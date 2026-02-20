FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
#RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt


# Copia SOLO il codice api dentro /app
COPY ./ /app/

EXPOSE 9999
CMD ["uvicorn", "src.api.endpoint:app", "--host", "0.0.0.0", "--port", "9999"]