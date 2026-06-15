from pydantic import BaseModel


class Article(BaseModel):
    id: str = ""
    title: str
    date: str
    url: str
    snippet: str
    body: str
