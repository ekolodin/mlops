"""Последовательно отправлять один валидный пример в локальный /predict."""

import argparse
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PASSENGER = {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
             "Parch": 0, "Fare": 7.25, "Embarked": "S"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--interval", type=float, default=0.15)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval < 0:
        parser.error("--seconds must be positive; --interval must be nonnegative")
    counts = {"ok": 0, "error": 0}
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        request = Request(args.url, data=json.dumps(PASSENGER).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=5) as response:
                json.load(response)
            counts["ok"] += 1
        except HTTPError as error:
            counts["error"] += 1
            detail = json.load(error).get("detail", {})
            print(json.dumps({"http_status": error.code, "detail": detail}), flush=True)
        # Other network errors remain visible: a stopped service is not a model timeout.
        time.sleep(args.interval)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
