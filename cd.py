import google.generativeai as genai
genai.configure(api_key="AIzaSyCzGqp92f4Ep2oLB0mbKXTPt7QcUm1KWZY")

for m in genai.list_models():
    print(m.name)
