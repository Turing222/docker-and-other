from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict():    
    return {"prediction": "This is a placeholder prediction."}
@app.get("/")
def read_root():
    return {"Hello": "MLOps Candidate", "Status": "Model Running"}