vectordb = []


def vector(input:str) -> list[float]:
    return [1.0,2.0,3.0]

def similarity(v1:list[float],v2:list[float])->float:
    if len(v1) != len(v1):
        return 0.0
    if len(v1) ==0 or len(v2)==0:
        return 0
    return 1.0

def chunk_text(text:str)->list[str]:
    return ["","",""]

def embed(vector_db:list,chunks:list[str]):    
    for chunk in chunks:
        vector = vector(chunk)
        vector_db.append({'chunk':'','vector':vector})