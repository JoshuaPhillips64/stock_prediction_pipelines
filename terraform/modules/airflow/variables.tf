# Variables for Airflow module
variable "subnet_ids" {
 type = list(string)
}
# terraform/modules/airflow/variables.tf
variable "airflow_config_dir" {
  type    = string
  default = "../../src/airflow_dags/config"
}

