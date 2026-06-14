from pydantic import BaseModel


class Article(BaseModel):
    title: str
    date: str
    url: str
    snippet: str
    body: str
