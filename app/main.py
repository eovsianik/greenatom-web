import sys

from django.conf import settings
from django.urls import path
from django.core.management import execute_from_command_line
from django.http import HttpResponse

from greenatom_web.predict import score_review

settings.configure(
    DEBUG=False,
    IGNORABLE_404_URLS=[r"^favicon\.ico$"],
    ROOT_URLCONF=sys.modules[__name__],
    ALLOWED_HOSTS=["*"],
)


def index(request):
    score = None
    score_classification = ""

    if request.method == "POST":
        review_text = request.POST.get("review_text", "")
        score = score_review(review_text)

        if score >= 5:
            score_classification = "Positive"
        else:
            score_classification = "Negative"

    return HttpResponse(f"""
        <html>
            <body>
                <h1>Score a movie review!</h1>
                <form method="post">
                    <textarea name="review_text" rows="4" cols="50" placeholder="Enter your review..."></textarea><br>
                    <button type="submit">Score Review</button>
                </form>
                <br>
                {'<h2>Score: {}</h2>'.format(score) if score is not None else ''}
                <h2>{score_classification}</h2>
            </body>
        </html>
    """)


urlpatterns = [
    path(r"", index),
]

if __name__ == "__main__":
    execute_from_command_line(sys.argv)
