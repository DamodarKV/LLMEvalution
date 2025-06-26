from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import os
from openai import OpenAI

#from GEval.GEval import expected_output_comprehensive

app = FastAPI()
templates = Jinja2Templates(directory="templates")



@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def form_post(
    request: Request,
    llm_input: str = Form(...),
    expected_output: str = Form(...),
    criteria: str = Form(...),
    metrics: str = Form(...)
):
    os.environ["OPENAI_API_KEY"] = "sk-..."  # your key here
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY not set")
    oai_client = OpenAI()


    try:
        response = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and creative assistant."},
                {"role": "user", "content": llm_input}
            ],
            temperature=0.7  # Optional: controls randomness (0.0 to 1.0)
        )
        actual_output = response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")



    # Create the metric
    metric = GEval(
        name=metrics,
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome="Poor"),
            Rubric(score_range=(3, 6), expected_outcome="Moderate"),
            Rubric(score_range=(7, 9), expected_outcome="Good"),
            Rubric(score_range=(10, 10), expected_outcome="Excellent")
        ],
        threshold=7
    )

    test_case = LLMTestCase(
        input=llm_input,
        actual_output=actual_output,
        expected_output=expected_output
    )

    evaluate([test_case], metrics=[metric])
    metric.measure(test_case)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "llm_input": llm_input,
        "expected_output": expected_output,
        "metrics": metrics,
        "criteria": criteria,
        "score": metric.score,
        "reason": metric.reason,
        "success": metric.success
    })

    #request,llm_input,expected_output,metrics,criteria,score,reason,success =
