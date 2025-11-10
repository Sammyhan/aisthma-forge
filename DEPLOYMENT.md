# AIsthma Forge Deployment Guide

## Local Deployment

### Quick Start

The simplest way to run AIsthma Forge locally is using the provided run script:

```bash
cd aisthma_forge
./run.sh
```

This script will activate the virtual environment, check dependencies, and launch the Streamlit application at `http://localhost:8501`.

### Manual Launch

If you prefer to launch manually:

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit
streamlit run app.py
```

### Configuration

Application settings can be customized in `.streamlit/config.toml`:

- **Theme colors**: Modify `primaryColor`, `backgroundColor`, etc.
- **Upload limits**: Adjust `maxUploadSize` (default: 200MB)
- **Port**: Change `port` if 8501 is already in use

## Cloud Deployment

### Streamlit Community Cloud

AIsthma Forge can be deployed for free on Streamlit Community Cloud:

1. Push your code to a GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and `app.py` as the main file
5. Click "Deploy"

The application will be available at `https://[your-app-name].streamlit.app`

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t aisthma-forge .
docker run -p 8501:8501 aisthma-forge
```

### AWS EC2 Deployment

1. Launch an EC2 instance (Ubuntu 22.04, t2.medium or larger)
2. SSH into the instance
3. Install dependencies:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv git
```

4. Clone repository and set up:

```bash
git clone https://github.com/yourusername/aisthma-forge.git
cd aisthma-forge
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. Run with nohup for persistent execution:

```bash
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
```

6. Configure security group to allow inbound traffic on port 8501

### Google Cloud Platform

Deploy using Cloud Run:

1. Create a `Dockerfile` (see above)
2. Build and push to Google Container Registry:

```bash
gcloud builds submit --tag gcr.io/[PROJECT-ID]/aisthma-forge
```

3. Deploy to Cloud Run:

```bash
gcloud run deploy aisthma-forge \
  --image gcr.io/[PROJECT-ID]/aisthma-forge \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi
```

### Azure App Service

1. Create an App Service with Python 3.11
2. Configure deployment from GitHub or local Git
3. Set startup command: `streamlit run app.py --server.port=8000 --server.address=0.0.0.0`
4. Adjust port in App Service settings to match

## Production Considerations

### Performance Optimization

For large datasets or high traffic, consider:

- **Caching**: Streamlit's `@st.cache_data` decorator is already used for expensive computations
- **Resource limits**: Increase memory allocation for cloud deployments (4GB minimum recommended)
- **Parallel processing**: Current implementation uses single-threaded analysis; consider multiprocessing for batch jobs

### Security

- **Authentication**: Add authentication layer for sensitive data (Streamlit supports custom auth)
- **HTTPS**: Always use HTTPS in production (automatic with Streamlit Cloud, configure for self-hosted)
- **Data privacy**: Ensure uploaded data is not logged or persisted beyond session
- **Input validation**: Application includes basic validation; review for production use

### Monitoring

- **Logs**: Streamlit logs to stdout; configure log aggregation for cloud deployments
- **Metrics**: Monitor CPU, memory, and request latency
- **Error tracking**: Integrate services like Sentry for error monitoring

### Backup and Recovery

- **Session state**: Current implementation uses in-memory session state; consider persistent storage for long-running analyses
- **Results export**: Users can download results; consider automatic backup to cloud storage
- **Database**: For multi-user deployments, consider adding PostgreSQL for storing analysis history

## Scaling

### Horizontal Scaling

For multiple concurrent users:

- Deploy behind a load balancer
- Use container orchestration (Kubernetes, ECS)
- Implement session affinity to maintain state

### Vertical Scaling

For large datasets:

- Increase instance memory (8GB+ for datasets with >1000 samples)
- Use compute-optimized instances for ML training
- Consider GPU instances for deep learning extensions

## Maintenance

### Updates

Keep dependencies updated:

```bash
pip list --outdated
pip install --upgrade [package-name]
pip freeze > requirements.txt
```

### Backups

Regularly backup:
- Application code (version control)
- Configuration files
- User data (if persisted)
- Analysis results (if stored)

### Monitoring

Set up alerts for:
- Application downtime
- High memory usage
- Error rate spikes
- Slow response times

## Troubleshooting

### Port Already in Use

If port 8501 is occupied:

```bash
streamlit run app.py --server.port=8502
```

### Memory Issues

For large datasets causing out-of-memory errors:

- Increase system memory
- Apply more aggressive filtering
- Process data in chunks
- Use sparse matrix representations

### Dependency Conflicts

If package conflicts occur:

```bash
# Create fresh environment
python3.11 -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

### Slow Performance

- Check dataset size (reduce features/samples if needed)
- Monitor CPU/memory usage
- Disable cross-validation for faster ML training
- Reduce SHAP calculation to top features only

## Support

For deployment assistance:

- GitHub Issues: Report bugs or request features
- Documentation: Refer to README and USER_GUIDE
- Community: Join discussions on GitHub Discussions

---

**Deployment Checklist:**

- [ ] Virtual environment created and activated
- [ ] Dependencies installed from requirements.txt
- [ ] Configuration customized in .streamlit/config.toml
- [ ] Application launches without errors
- [ ] Example dataset loads successfully
- [ ] All analysis modules functional
- [ ] Results export working
- [ ] Security measures implemented (if production)
- [ ] Monitoring configured (if production)
- [ ] Backup strategy in place (if production)

🫁 **Ready to deploy AIsthma Forge and accelerate asthma research!**
