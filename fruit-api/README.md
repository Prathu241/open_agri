# Jeevandhara fruit disease API

Standalone Docker service for the calibrated `public-fruit-v1` research model.

## What it does

- accepts JPEG, PNG, or WebP images up to 10 MB;
- rejects very small, dark, overexposed, featureless, or obviously blurred photos;
- returns calibrated top-three crop-condition candidates;
- returns `uncertain` below the validated confidence threshold;
- rejects a result when it conflicts with the optional farmer-selected crop;
- never returns treatment or pesticide instructions.

Supported crops: apple, banana, blueberry, cherry, grape, peach, and strawberry.

## Local run

```powershell
docker build -t jeevandhara-fruit-api fruit-api
docker run --rm -p 8002:8000 jeevandhara-fruit-api
```

Open `http://127.0.0.1:8002/health` and `http://127.0.0.1:8002/docs`.

`POST /v1/predict` uses multipart fields:

- `file`: required image;
- `expected_crop`: optional supported crop name.

## Hosting

Deploy the `fruit-api` directory as a Docker service with at least 2 GB RAM and `/health` as its health-check path. After it has an HTTPS URL, set that URL as `CROP_HEALTH_API_URL` in the Jeevandhara Vercel project and redeploy the frontend.

This model is research-only. Its public training data includes non-commercially licensed PlantWild data, and it has not been validated on Jeevandhara farm captures.
