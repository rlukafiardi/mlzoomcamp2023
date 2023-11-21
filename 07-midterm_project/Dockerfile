FROM python:3.11.6-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "xgboost_model.bin", "./"]

EXPOSE 1313

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:1313", "predict:app"]
