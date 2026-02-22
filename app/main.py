from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <body>
        <h1>🐍 pdf_llm_app 起動成功</h1>
        <h2>バージョン</h2>
        <p>02221518</p>
        <p>02221734</p>
      </body>
    </html>
    """
