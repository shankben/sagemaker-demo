from sagemaker.huggingface import HuggingFace


class SageMakerDemoHelper:
    _instance = None
    job_name = "imdb-huggingface"
    
    metric_definitions = [
        {
          "Name": "loss", 
          "Regex": "'loss': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "learning_rate", 
          "Regex": "'learning_rate': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "eval_loss", 
          "Regex": "'eval_loss': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "eval_accuracy", 
          "Regex": "'eval_accuracy': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "eval_f1", 
          "Regex": "'eval_f1': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "eval_precision", 
          "Regex": "'eval_precision': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "eval_recall", 
          "Regex": "'eval_recall': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "eval_runtime", 
          "Regex": "'eval_runtime': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "eval_samples_per_second", 
          "Regex": "'eval_samples_per_second': ([0-9]+(.|e\-)[0-9]+),?"
        },
        {
          "Name": "epoch", 
          "Regex": "'epoch': ([0-9]+(.|e\-)[0-9]+),?"
        }
    ]
   
    @staticmethod
    def instance(bucket, role, model_name):
        if SageMakerDemoHelper._instance is None:
            SageMakerDemoHelper(bucket, role, model_name)
        return SageMakerDemoHelper._instance

    def __init__(self, bucket, role, model_name):
        if SageMakerDemoHelper._instance is not None:
            raise Exception("This class is a singleton!")

        SageMakerDemoHelper._instance = self
        
        self.bucket = bucket
        self.role = role
        self.model_name = model_name
        
        self.params = {
            "base_job_name": self.job_name,
            "enable_sagemaker_metrics": True,
            "entry_point": "train.py",
            "instance_count": 1,
            "instance_type": "ml.p3.16xlarge",
            "py_version": "py36",
            "pytorch_version": "1.6.0",
            "role": self.role,
            "source_dir": "./scripts",
            "transformers_version": "4.4.2"
        }

        self.spot_params = {
            "checkpoint_s3_uri": f"s3://{self.bucket}/{self.job_name}/checkpoints",
            "use_spot_instances": True,
            "max_wait": 3600,
            "max_run": 3600
        }

        self.dataparallel_params = {
            "instance_count": 2,
            "distribution": {
                "smdistributed": {
                    "dataparallel": {
                        "enabled": True
                    }
                },
                "mpi": {
                    "enabled": True,
                    "processes_per_host" : 2
                }
            }
        }

        self.modelparallel_params = {
            "instance_count": 2,
            "distribution": {
                "smdistributed": {
                    "modelparallel": {
                        "enabled": True,
                        "parameters": {
                            "microbatches": 4,
                            "placement_strategy": "spread",
                            "pipeline": "interleaved",
                            "optimize": "speed",
                            "partitions": 4,
                            "ddp": True
                        }
                    }
                },
                "mpi": {
                    "enabled": True,
                    "processes_per_host" : 2
                }
            }
        }

        self.hyperparams = {
            "epochs": 3,
            "eval_batch_size": 128,
            "model_name": self.model_name,
            "train_batch_size": 64
        }

    def use_standard_training(self):
        return HuggingFace(
            **self.params,
            hyperparameters = self.hyperparams,
            metric_definitions = self.metric_definitions
        )

    def use_spot(self):
        return HuggingFace(
            **self.params,
            **self.spot_params,
            metric_definitions = self.metric_definitions,
            hyperparameters = {
                **self.hyperparams,
                "output_dir": "/opt/ml/checkpoints"
            }
        )

    def use_spot_distributed(self):
        return HuggingFace(**{
            **self.params,
            **self.spot_params, 
            **self.distributed_params,
            "metric_definitions": self.metric_definitions,
            "hyperparameters": {
                **self.hyperparams,
                "output_dir": "/opt/ml/checkpoints"
            }
        })

    def use_distributed(self):
        return HuggingFace(**{
            **self.params,
            **self.distributed_params,
            "metric_definitions": self.metric_definitions
        })

