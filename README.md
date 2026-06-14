# News

## Setup
1. `python -m venv .venv`
2. activate .venv
3. pip install -r requirements.txt
4. create .env from .env.example
5. in mysql run `CREATE DATABASE IF NOT EXISTS news_agent;`
6. Add GUARDIAN_API_KEY (.env) -> visit `https://bonobo.capi.gutools.co.uk/register/developer`

## Migrations
- apply: `alembic upgrade head`
- make: `alembic revision --autogenerate -m "example_name"`