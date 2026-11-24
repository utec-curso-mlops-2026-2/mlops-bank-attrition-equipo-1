"""Despliegue del batch en AWS (R7): SageMaker Model + Pipeline + EventBridge Scheduler. Lo usa `cd.yml`.

    EventBridge Scheduler (mensual) ──► SageMaker Pipeline `equipo1-bank-attrition-batch`
        1. PrepararFeatures   (Processing)       python -m bank_attrition.container prepare
        2. ScorearTransform   (Batch Transform)  contenedor `serve` con el modelo del Registry por alias
        3. PublicarScoresDrift (Processing)      python -m bank_attrition.container publish
    Logs: CloudWatch /aws/sagemaker/ProcessingJobs y /aws/sagemaker/TransformJobs.

Uso:
    python -m bank_attrition.aws_batch deploy --image-uri <ecr@sha256:...> --git-sha <sha>
    python -m bank_attrition.aws_batch run [--wait]                 # ejecución manual
    python -m bank_attrition.aws_batch schedule-once --minutes 5   # evidencia de ejecución automática
    python -m bank_attrition.aws_batch status [--execution-arn ...]
Nombres, roles y bucket se leen de variables de entorno (ver docs/batch_sagemaker.md).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

from botocore.exceptions import ClientError

PIPELINE_SCHEMA = "2020-12-01"


def settings() -> dict:
    env = os.environ
    return {
        "region": env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-1",
        "pipeline": env.get("BATCH_PIPELINE_NAME", "equipo1-bank-attrition-batch"),
        "model_prefix": env.get("BATCH_MODEL_PREFIX", "equipo1-bank-attrition"),
        "role_arn": env["SAGEMAKER_ROLE_ARN"],
        "scheduler_role_arn": env.get("SCHEDULER_ROLE_ARN", ""),
        "bucket": env["MLOPS_DATA_BUCKET"],
        "release": env.get("BATCH_RELEASE", "202408"),
        "alias": env.get("BATCH_MODEL_ALIAS", "champion"),
        "schedule": env.get("BATCH_SCHEDULE_NAME", "equipo1-batch-mensual"),
        "cron": env.get("BATCH_SCHEDULE_CRON", "cron(0 6 1 * ? *)"),  # día 1 de cada mes, 06:00
        "timezone": env.get("BATCH_SCHEDULE_TZ", "America/Lima"),
        "processing_instance": env.get("BATCH_PROCESSING_INSTANCE", "ml.t3.medium"),
        "transform_instance": env.get("BATCH_TRANSFORM_INSTANCE", "ml.m5.large"),
        "mlflow_uri": env["MLFLOW_TRACKING_URI"],
        "mlflow_experiment": env.get("MLFLOW_EXPERIMENT_NAME", "equipo-1-fuga"),
        "mlflow_user": env.get("MLFLOW_TRACKING_USERNAME", "admin"),
        "mlflow_password_ssm": env.get("MLFLOW_TRACKING_PASSWORD_SSM", "/equipo1/mlflow/admin-password"),
    }


def container_env(s: dict) -> dict[str, str]:
    """Variables del contenedor. La contraseña de MLflow NO está aquí: se lee de SSM en tiempo de ejecución."""
    return {
        "AWS_DEFAULT_REGION": s["region"],
        "MLOPS_DATA_SOURCE": "s3",
        "MLOPS_DATA_BUCKET": s["bucket"],
        "MLFLOW_TRACKING_URI": s["mlflow_uri"],
        "MLFLOW_EXPERIMENT_NAME": s["mlflow_experiment"],
        "MLFLOW_TRACKING_USERNAME": s["mlflow_user"],
        "MLFLOW_TRACKING_PASSWORD_SSM": s["mlflow_password_ssm"],
        "USER_NAME": "sagemaker-pipeline",
    }


def _join(*values) -> dict:
    return {"Std:Join": {"On": "/", "Values": list(values)}}


def _param(name: str) -> dict:
    return {"Get": f"Parameters.{name}"}


EXECUTION_ID = {"Get": "Execution.PipelineExecutionId"}


def _processing_step(name: str, command: list, s: dict, image_uri: str, depends_on: list[str]) -> dict:
    step = {
        "Name": name,
        "Type": "Processing",
        "Arguments": {
            "ProcessingResources": {"ClusterConfig": {
                "InstanceType": _param("ProcessingInstanceType"), "InstanceCount": 1, "VolumeSizeInGB": 10}},
            "AppSpecification": {"ImageUri": image_uri, "ContainerArguments": command},
            "RoleArn": s["role_arn"],
            "Environment": container_env(s),
            "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        },
    }
    if depends_on:
        step["DependsOn"] = depends_on
    return step


def pipeline_definition(s: dict, image_uri: str, model_name: str, batch_prefix: str = "batch",
                        input_file: str = "registros.jsonl") -> dict:
    """Definición JSON del SageMaker Pipeline (sin SDK de SageMaker: solo boto3)."""
    base = _join(f"s3://{s['bucket']}/{batch_prefix}", _param("Release"), EXECUTION_ID)
    run_prefix = base["Std:Join"]["Values"]
    common = ["--release", _param("Release"), "--execution-id", EXECUTION_ID]
    prepare = _processing_step("PrepararFeatures", ["prepare", *common, "--alias", _param("ModelAlias")],
                               s, image_uri, [])
    transform = {
        "Name": "ScorearTransform",
        "Type": "Transform",
        "DependsOn": ["PrepararFeatures"],
        "Arguments": {
            "ModelName": model_name,
            "BatchStrategy": "MultiRecord",
            "MaxPayloadInMB": 6,
            "MaxConcurrentTransforms": 1,
            "Environment": {"MODEL_ALIAS": _param("ModelAlias")},
            "TransformInput": {
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": _join(*run_prefix, "input")}},
                "ContentType": "application/jsonlines",
                "SplitType": "Line",
            },
            "TransformOutput": {"S3OutputPath": _join(*run_prefix, "output"), "Accept": "application/jsonlines",
                                "AssembleWith": "Line"},
            "TransformResources": {"InstanceType": _param("TransformInstanceType"), "InstanceCount": 1},
        },
    }
    publish = _processing_step("PublicarScoresDrift", ["publish", *common], s, image_uri, ["ScorearTransform"])
    return {
        "Version": PIPELINE_SCHEMA,
        "Metadata": {},
        "Parameters": [
            {"Name": "Release", "Type": "String", "DefaultValue": s["release"]},
            {"Name": "ModelAlias", "Type": "String", "DefaultValue": s["alias"]},
            {"Name": "ProcessingInstanceType", "Type": "String", "DefaultValue": s["processing_instance"]},
            {"Name": "TransformInstanceType", "Type": "String", "DefaultValue": s["transform_instance"]},
        ],
        "Steps": [prepare, transform, publish],
    }


def model_name_for(s: dict, git_sha: str) -> str:
    return f"{s['model_prefix']}-{git_sha[:12]}"[:63]


def _clients(region: str):
    import boto3

    return boto3.client("sagemaker", region_name=region), boto3.client("scheduler", region_name=region)


def ensure_model(sm, s: dict, image_uri: str, git_sha: str) -> str:
    """SageMaker Model que usa el Batch Transform: imagen por digest + entorno (sin secretos)."""
    name = model_name_for(s, git_sha)
    try:
        current = sm.describe_model(ModelName=name)["PrimaryContainer"]["Image"]
        if current != image_uri:
            raise RuntimeError(f"El Model {name} ya existe con otra imagen ({current}); use otro commit")
        return name
    except ClientError as exc:
        if "Could not find model" not in str(exc):
            raise
    env = {**container_env(s), "MODEL_ALIAS": s["alias"]}
    sm.create_model(ModelName=name, ExecutionRoleArn=s["role_arn"],
                    PrimaryContainer={"Image": image_uri, "Environment": env},
                    Tags=[{"Key": "project", "Value": "equipo1-bank-attrition"}, {"Key": "git_sha", "Value": git_sha}])
    return name


def upsert_pipeline(sm, s: dict, definition: dict, git_sha: str) -> str:
    body = json.dumps(definition)
    args = {"PipelineName": s["pipeline"], "PipelineDefinition": body, "RoleArn": s["role_arn"],
            "PipelineDescription": f"Batch mensual de fuga (R7): preparar -> Batch Transform -> publicar. commit {git_sha[:12]}"}
    try:
        sm.describe_pipeline(PipelineName=s["pipeline"])
        return sm.update_pipeline(**args)["PipelineArn"]
    except sm.exceptions.ResourceNotFound:
        return sm.create_pipeline(**args, Tags=[{"Key": "project", "Value": "equipo1-bank-attrition"}])["PipelineArn"]


def _target(s: dict, pipeline_arn: str) -> dict:
    return {"Arn": pipeline_arn, "RoleArn": s["scheduler_role_arn"],
            "SageMakerPipelineParameters": {"PipelineParameterList": [
                {"Name": "Release", "Value": s["release"]}, {"Name": "ModelAlias", "Value": s["alias"]}]},
            "RetryPolicy": {"MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 3600}}


def upsert_schedule(scheduler, s: dict, pipeline_arn: str) -> str:
    args = {"Name": s["schedule"], "ScheduleExpression": s["cron"], "ScheduleExpressionTimezone": s["timezone"],
            "FlexibleTimeWindow": {"Mode": "OFF"}, "Target": _target(s, pipeline_arn), "State": "ENABLED",
            "Description": "Batch mensual de scores de fuga (R7)"}
    try:
        scheduler.get_schedule(Name=s["schedule"])
        return scheduler.update_schedule(**args)["ScheduleArn"]
    except scheduler.exceptions.ResourceNotFoundException:
        return scheduler.create_schedule(**args)["ScheduleArn"]


def schedule_once(scheduler, s: dict, pipeline_arn: str, minutes: int) -> dict:
    """Schedule de una sola vez (como la S11) para evidenciar una ejecución automática."""
    when = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).replace(second=0, microsecond=0)
    name = f"{s['schedule']}-evidencia-{when:%Y%m%d%H%M}"
    scheduler.create_schedule(Name=name, ScheduleExpression=f"at({when:%Y-%m-%dT%H:%M:%S})",
                              ScheduleExpressionTimezone="UTC", FlexibleTimeWindow={"Mode": "OFF"},
                              Target=_target(s, pipeline_arn), ActionAfterCompletion="NONE",
                              Description="Ejecución automática de evidencia del batch (R7)")
    return {"schedule": name, "utc": when.isoformat()}


def execution_status(sm, execution_arn: str, wait: bool = False, poll: int = 30) -> dict:
    while True:
        desc = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        status = desc["PipelineExecutionStatus"]
        if not wait or status not in {"Executing", "Stopping"}:
            break
        time.sleep(poll)
    steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn, SortOrder="Ascending")
    return {"execution_arn": execution_arn, "status": status,
            "failure_reason": desc.get("FailureReason"),
            "steps": [{"step": st["StepName"], "status": st["StepStatus"],
                       "job": next(iter(st.get("Metadata", {}).values()), {}).get("Arn"),
                       "failure": st.get("FailureReason")} for st in steps["PipelineExecutionSteps"]]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    dep = sub.add_parser("deploy", help="Model + Pipeline + schedule mensual")
    dep.add_argument("--image-uri", required=True)
    dep.add_argument("--git-sha", required=True)
    dep.add_argument("--no-schedule", action="store_true")
    run = sub.add_parser("run", help="ejecución manual del pipeline")
    run.add_argument("--wait", action="store_true")
    once = sub.add_parser("schedule-once", help="ejecución automática de evidencia")
    once.add_argument("--minutes", type=int, default=5)
    st = sub.add_parser("status")
    st.add_argument("--execution-arn", help="por defecto, la última ejecución")
    st.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)

    s = settings()
    sm, scheduler = _clients(s["region"])
    if args.cmd == "deploy":
        model = ensure_model(sm, s, args.image_uri, args.git_sha)
        arn = upsert_pipeline(sm, s, pipeline_definition(s, args.image_uri, model), args.git_sha)
        out = {"model_name": model, "pipeline_arn": arn}
        if not args.no_schedule:
            out["schedule_arn"] = upsert_schedule(scheduler, s, arn)
    elif args.cmd == "run":
        arn = sm.start_pipeline_execution(
            PipelineName=s["pipeline"], PipelineExecutionDisplayName=f"manual-{datetime.now(timezone.utc):%Y%m%d%H%M%S}",
            PipelineParameters=[{"Name": "Release", "Value": s["release"]},
                                {"Name": "ModelAlias", "Value": s["alias"]}])["PipelineExecutionArn"]
        out = execution_status(sm, arn, wait=args.wait)
    elif args.cmd == "schedule-once":
        arn = sm.describe_pipeline(PipelineName=s["pipeline"])["PipelineArn"]
        out = schedule_once(scheduler, s, arn, args.minutes)
    else:
        execution = args.execution_arn or sm.list_pipeline_executions(
            PipelineName=s["pipeline"], SortBy="CreationTime", SortOrder="Descending",
            MaxResults=1)["PipelineExecutionSummaries"][0]["PipelineExecutionArn"]
        out = execution_status(sm, execution, wait=args.wait)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
