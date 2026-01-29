FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit","run","app/streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]
