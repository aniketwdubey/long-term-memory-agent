"""The deployed stack: the same container and the same Postgres, run by AWS.

The design goal here is that **nothing about the application changes**. The
container is the image `docker compose` already builds, and the database is the
same PostgreSQL with the same pgvector extension the local container runs — so
``PostgresStore`` and ``PostgresSaver`` are the identical, official LangGraph
classes the 237 offline tests exercise. There is no cloud-only storage adapter,
which means the deployed system is the tested system rather than a sibling of it.

That property is why this is RDS and not a vector database. Postgres does two
jobs for this project — the memory store *and* the thread checkpointer — and any
vector-native replacement (S3 Vectors, OpenSearch) covers only the first, leaving
transcripts homeless and requiring a hand-written ``BaseStore`` on top. Once a
relational store has to exist anyway, pgvector comes with it for free.

Compute is **ECS Express Mode**, AWS's successor to App Runner (sunset April
2026). It takes a container image and returns a managed HTTPS endpoint with
autoscaling, which is the whole requirement. Lambda was rejected deliberately:
this app holds a Postgres connection pool for its process lifetime, and Lambda's
model would rebuild that pool on every cold start and multiply connections
against the instance limit under concurrency.

Everything is `RemovalPolicy.DESTROY`. `make destroy` should leave nothing behind.
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_rds as rds
from constructs import Construct

# Express Mode exposes no runtimePlatform, so tasks run x86_64. The image must
# be built to match: project 07 lost three deploys to an arm64 image on an
# x86 runtime failing with "exec format error", and this is the same trap from
# the other direction.
IMAGE_PLATFORM = ecr_assets.Platform.LINUX_AMD64

CONTAINER_PORT = 8000
DB_NAME = "engram"
DB_USER = "engram"


class EngramStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        chat_model: str = "amazon.nova-lite-v1:0",
        reasoning_model: str = "amazon.nova-pro-v1:0",
        embed_model: str = "amazon.titan-embed-text-v2:0",
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)  # type: ignore[arg-type]

        # --- network ------------------------------------------------------
        # One NAT rather than a set of interface endpoints: the task needs ECR,
        # CloudWatch Logs, Secrets Manager and Bedrock, and a single managed
        # route to all four has fewer ways to fail on a first deploy than five
        # endpoints do. Swapping to endpoints-only is the hardening step, and it
        # buys a real property — no internet egress from a service holding
        # personal data — so it is worth doing once this is proven.
        vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=1)

        db_sg = ec2.SecurityGroup(self, "DatabaseSg", vpc=vpc, allow_all_outbound=False)
        app_sg = ec2.SecurityGroup(self, "AppSg", vpc=vpc, allow_all_outbound=True)
        db_sg.add_ingress_rule(app_sg, ec2.Port.tcp(5432), "engram app to postgres")

        # --- database -----------------------------------------------------
        database = rds.DatabaseInstance(
            self,
            "Database",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_17_6
            ),
            # Burstable Graviton: this workload is small and bursty, and the
            # instance class is unrelated to the container's architecture.
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE4_GRAVITON, ec2.InstanceSize.SMALL
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[db_sg],
            # RDS generates the password and stores it in Secrets Manager; it is
            # never in the template, the environment, or this repository.
            credentials=rds.Credentials.from_generated_secret(DB_USER),
            database_name=DB_NAME,
            allocated_storage=20,
            # Free, and there is no argument for storing someone's memories
            # unencrypted at rest.
            storage_encrypted=True,
            max_allocated_storage=50,
            multi_az=False,
            backup_retention=cdk.Duration.days(0),
            delete_automated_backups=True,
            deletion_protection=False,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        assert database.secret is not None

        # --- image --------------------------------------------------------
        image = ecr_assets.DockerImageAsset(
            self,
            "Image",
            directory="..",
            platform=IMAGE_PLATFORM,
            exclude=["infra", ".venv", ".git", "data", "docs"],
        )

        log_group = logs.LogGroup(
            self,
            "Logs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # --- roles --------------------------------------------------------
        execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )
        # Pulls the password at task start so it reaches the container as an
        # environment variable without ever passing through the template.
        database.secret.grant_read(execution_role)
        log_group.grant_write(execution_role)

        task_role = iam.Role(
            self, "TaskRole", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    # Converse-style clients stream; 07 learned this the hard way
                    # with a 403 on first deploy when only InvokeModel was granted.
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.*",
                    f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/*",
                ],
            )
        )
        log_group.grant_write(task_role)

        # The role ECS itself assumes to manage the gateway in front of the task.
        infrastructure_role = iam.Role(
            self,
            "InfrastructureRole",
            assumed_by=iam.ServicePrincipal("ecs.amazonaws.com"),
            managed_policies=[
                # Under service-role/, not the root path. The first deploy
                # failed on exactly this: the policy NAME was confirmed by
                # enumerating IAM, but not its ARN, and the path is part of it.
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"
                )
            ],
        )

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        # --- the service --------------------------------------------------
        secret_arn = database.secret.secret_arn
        service = ecs.CfnExpressGatewayService(
            self,
            "Service",
            infrastructure_role_arn=infrastructure_role.role_arn,
            execution_role_arn=execution_role.role_arn,
            task_role_arn=task_role.role_arn,
            cluster=cluster.cluster_arn,
            service_name="engram",
            health_check_path="/health",
            cpu="1024",
            memory="2048",
            # ECS manages an ALB in front of the task (see the ingress_path_*
            # attributes), and it places that load balancer in the subnets given
            # here. Public subnets therefore, or the managed endpoint comes up
            # internal and the whole point of deploying — a URL you can open —
            # is lost. This is the one assumption in the stack not verified
            # against a real deploy; if the endpoint resolves but does not
            # answer, this is the first thing to check.
            network_configuration=ecs.CfnExpressGatewayService.ExpressGatewayServiceNetworkConfigurationProperty(
                subnets=vpc.select_subnets(subnet_type=ec2.SubnetType.PUBLIC).subnet_ids,
                security_groups=[app_sg.security_group_id],
            ),
            scaling_target=ecs.CfnExpressGatewayService.ExpressGatewayScalingTargetProperty(
                min_task_count=1,
                max_task_count=2,
                auto_scaling_metric="REQUEST_COUNT_PER_TARGET",
                auto_scaling_target_value=50,
            ),
            primary_container=ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
                image=image.image_uri,
                container_port=CONTAINER_PORT,
                command=[
                    "uvicorn",
                    "engram.api.main:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(CONTAINER_PORT),
                ],
                aws_logs_configuration=ecs.CfnExpressGatewayService.ExpressGatewayServiceAwsLogsConfigurationProperty(
                    log_group=log_group.log_group_name,
                    log_stream_prefix="engram",
                ),
                environment=[
                    _env("ENGRAM_STORE_BACKEND", "postgres"),
                    _env("ENGRAM_POSTGRES_HOST", database.db_instance_endpoint_address),
                    _env("ENGRAM_POSTGRES_PORT", database.db_instance_endpoint_port),
                    _env("ENGRAM_POSTGRES_USER", DB_USER),
                    _env("ENGRAM_POSTGRES_DB", DB_NAME),
                    # The deployed service runs the real thing; the stub and the
                    # hashing embedder stay the default everywhere else.
                    _env("ENGRAM_CHAT_PROVIDER", "bedrock"),
                    _env("ENGRAM_EMBEDDER", "bedrock"),
                    _env("ENGRAM_AWS_REGION", self.region),
                    _env("ENGRAM_BEDROCK_MODEL_ID", chat_model),
                    _env("ENGRAM_BEDROCK_REASONING_MODEL_ID", reasoning_model),
                    _env("ENGRAM_BEDROCK_EMBED_MODEL_ID", embed_model),
                    _env("ENGRAM_LOG_FORMAT", "json"),
                    # Spans to the container logs, where CloudWatch collects
                    # them. Building tracing and then leaving it off in the one
                    # environment nobody can attach a debugger to would be an
                    # odd place to economise.
                    _env("ENGRAM_OTEL_EXPORTER", "console"),
                    _env("ENGRAM_OTEL_SERVICE_NAME", "engram"),
                ],
                secrets=[
                    ecs.CfnExpressGatewayService.SecretProperty(
                        name="ENGRAM_POSTGRES_PASSWORD",
                        value_from=f"{secret_arn}:password::",
                    )
                ],
            ),
        )
        service.node.add_dependency(database)

        cdk.CfnOutput(self, "ServiceName", value="engram")
        cdk.CfnOutput(self, "DatabaseEndpoint", value=database.db_instance_endpoint_address)
        cdk.CfnOutput(self, "DatabaseSecretArn", value=secret_arn)
        cdk.CfnOutput(self, "LogGroup", value=log_group.log_group_name)
        cdk.CfnOutput(self, "ServiceUrl", value=service.attr_endpoint)


def _env(name: str, value: str) -> ecs.CfnExpressGatewayService.KeyValuePairProperty:
    return ecs.CfnExpressGatewayService.KeyValuePairProperty(name=name, value=value)
