FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict_web.py", "model1.bin", "./"]

EXPOSE 1313

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:1313", "predict:app"]