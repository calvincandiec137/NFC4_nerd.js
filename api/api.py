from fastapi import FastAPI
from modules.response import main


app=FastAPI()

@app.post('/')
def test(query):   
   return main(query)

