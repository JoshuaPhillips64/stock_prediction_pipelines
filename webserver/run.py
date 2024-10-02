from app import create_app, db

app = create_app()  # Create the app instance

# Import AiStockPredictions from the app instance
AiStockPredictions = app.AiStockPredictions

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'EnrichedStockData': EnrichedStockData, 'AiStockPredictions': AiStockPredictions}

if __name__ == '__main__':
    app.run(port=7237, debug=True)