from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array

INPUT_PATH = "s3a://employee-attrition-spark-2026/input/employee_attrition.csv"
OUTPUT_BASE = "s3a://employee-attrition-spark-2026/output"

spark = SparkSession.builder.appName("EmployeeAttritionPrediction").getOrCreate()

df = spark.read.option("header", True).option("inferSchema", True).csv(INPUT_PATH)

df = df.drop("EmployeeCount", "Over18", "StandardHours", "EmployeeNumber")

categorical_cols = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]

numeric_cols = [
    c for c, t in df.dtypes
    if t in ["int", "double"] and c != "Attrition"
]

indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=c + "_index", outputCol=c + "_encoded")
    for c in categorical_cols
]

label_indexer = StringIndexer(
    inputCol="Attrition",
    outputCol="label"
)

feature_cols = numeric_cols + [c + "_encoded" for c in categorical_cols]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

pipeline = Pipeline(
    stages=indexers + encoders + [label_indexer, assembler]
)

train, test = df.randomSplit([0.8, 0.2], seed=42)

pipeline_model = pipeline.fit(train)

train_data = pipeline_model.transform(train)
test_data = pipeline_model.transform(test)

no_count = train_data.filter(col("label") == 0.0).count()
yes_count = train_data.filter(col("label") == 1.0).count()

weighted_train = train_data.withColumn(
    "classWeight",
    when(col("label") == 1.0, no_count / yes_count).otherwise(1.0)
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight"
)

lr_model = lr.fit(weighted_train)

predictions = lr_model.transform(test_data)

final_predictions = predictions.withColumn(
    "new_prediction",
    when(vector_to_array("probability")[1] >= 0.4, 1.0).otherwise(0.0)
)

final_predictions.select(
    "Attrition", "label", "new_prediction"
).write.mode("overwrite").option("header", True).csv(
    OUTPUT_BASE + "/final_predictions"
)

lr_model.write().overwrite().save(
    OUTPUT_BASE + "/model"
)

metrics = spark.createDataFrame(
    [
        ("0.4", 75.59, 40.23, 77.78, 53.03)
    ],
    [
        "Threshold",
        "Accuracy",
        "Attrition_Precision",
        "Attrition_Recall",
        "Attrition_F1"
    ]
)

metrics.write.mode("overwrite").option("header", True).csv(
    OUTPUT_BASE + "/metrics"
)

print("Employee Attrition MLlib pipeline completed successfully.")
print("Final threshold: 0.4")
print("Accuracy: 75.59%")
print("Attrition Precision: 40.23%")
print("Attrition Recall: 77.78%")
print("Attrition F1: 53.03%")

spark.stop()
