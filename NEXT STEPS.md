#Next Steps

###1. Reconfigure the SARIMAX and Binary Model:
- Support for input from hyperparameter
- Support for input from different feature sets
- Support for input on different shift period/prediction horizon
- Make unique key for each model as symbol, shift period, and prediction horizon
- Check the database for the latest data, if not there use Alpha Vantage

###2. Reconfigure the Make Predictions function:
- Should just take in the unique key, look it up in S3, and run the function

###3. Implement the Model in the Web App:
- Plug in the new pages in templates into routes. Figure out how to make front end kick off lambda.
- Generate a loading bar, other visualizations, and error messages

###4. Lambda and Airflow Setup:
- Make the train_model function a lambda function
- Make the make_predictions function a lambda function
- Make the ingest_data function a lambda function
- Make the ai_analysis review of output a lambda function
- Set up Airflow to load the top 50 daily stocks each day.