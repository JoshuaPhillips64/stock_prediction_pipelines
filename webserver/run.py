from app import create_app, db
from app.models import EnrichedStockData

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'EnrichedStockData': EnrichedStockData}

if __name__ == '__main__':
    app.run(port=7237, debug=True)