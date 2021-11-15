FROM python:3.8
EXPOSE 8501
COPY . /ml_app
WORKDIR /ml_app
RUN pip install -r requirements.txt
CMD streamlit run app.py