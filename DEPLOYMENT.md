# DoChat Deployment Guide for Render

## 🆓 100% FREE Stack

This guide will help you deploy DoChat to Render using completely free services:

- **Embeddings:** Google Gemini text-embedding-004 (FREE)
- **Vector DB:** Qdrant Cloud (1GB FREE forever)
- **LLM:** Groq (FREE tier)
- **Hosting:** Render (FREE tier)

---

## Prerequisites

### 1. Get Your API Keys (All FREE!)

#### Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

#### Groq API Key
1. Go to [Groq Console](https://console.groq.com/keys)
2. Sign up for free account
3. Create a new API key
4. Copy your API key

#### Qdrant Cloud (Optional but recommended for production)
1. Go to [Qdrant Cloud](https://cloud.qdrant.io/)
2. Sign up for free account
3. Create a new cluster (1GB free)
4. Copy your cluster URL and API key

---

## Deployment Steps

### Option 1: Deploy via Render Dashboard (Easiest)

1. **Fork/Push this repository to GitHub**

2. **Go to [Render Dashboard](https://dashboard.render.com/)**

3. **Create a New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the `DoChat` repository

4. **Configure the Service**
   - **Name:** `dochat` (or your preferred name)
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn modules.app:app --host 0.0.0.0 --port $PORT`

5. **Add Environment Variables**
   Click "Advanced" and add:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   QDRANT_URL=:memory:
   ```
   
   **For Production with Qdrant Cloud:**
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   QDRANT_URL=https://your-cluster.cloud.qdrant.io
   QDRANT_API_KEY=your_qdrant_api_key_here
   ```

6. **Add Persistent Disk (Optional)**
   - For uploaded PDFs to persist across deploys
   - Name: `dochat-storage`
   - Mount Path: `/opt/render/project/src/database`
   - Size: 1GB (free tier)

7. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete

---

### Option 2: Deploy via render.yaml (Automated)

The project includes a `render.yaml` file for automated deployment:

1. **Go to [Render Dashboard](https://dashboard.render.com/)**

2. **Create New Blueprint Instance**
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Select the `DoChat` repository
   - Render will automatically detect `render.yaml`

3. **Set Environment Variables in Render Dashboard**
   - `GEMINI_API_KEY`
   - `GROQ_API_KEY`
   - `QDRANT_URL` (optional, defaults to `:memory:`)
   - `QDRANT_API_KEY` (only if using Qdrant Cloud)

4. **Apply Blueprint**

---

## Testing Your Deployment

Once deployed, test the endpoints:

1. **Health Check**
   ```
   GET https://your-app.onrender.com/
   ```
   Should return: `{"message": "Rag is Running"}`

2. **Upload a PDF**
   ```bash
   curl -X POST https://your-app.onrender.com/upload \
     -H "Content-Type: application/json" \
     -d '{
       "data": "base64_encoded_pdf_here",
       "name": "test.pdf"
     }'
   ```

3. **Prepare/Ingest Documents**
   ```
   GET https://your-app.onrender.com/prepare
   ```

4. **Ask a Question**
   ```
   GET https://your-app.onrender.com/ask?q=What is this document about?
   ```

---

## Important Notes

### Free Tier Limitations

**Render Free Tier:**
- ✅ Spins down after 15 min of inactivity
- ✅ 750 hours/month free
- ⚠️ Cold starts take 30-60 seconds

**Qdrant Cloud Free Tier:**
- ✅ 1GB storage (100,000+ vectors)
- ✅ Always on
- ✅ No credit card required

**Google Gemini Free Tier:**
- ✅ 15 requests/min
- ✅ 1,500 requests/day
- ✅ Good for development and small apps

**Groq Free Tier:**
- ✅ 30 requests/min
- ✅ Very fast inference
- ✅ Multiple models available

### Using In-Memory Qdrant vs Cloud

**In-Memory (`:memory:`):**
- ✅ No setup needed
- ⚠️ Data lost on restart
- ✅ Good for testing
- Set: `QDRANT_URL=:memory:`

**Qdrant Cloud:**
- ✅ Persistent storage
- ✅ Production-ready
- ✅ 1GB free forever
- Set: `QDRANT_URL=https://your-cluster.cloud.qdrant.io`
- Set: `QDRANT_API_KEY=your_key`

### Persistent File Storage

By default, uploaded PDFs are stored in the `./database/` directory, which is ephemeral on Render's free tier.

**To make PDFs persistent:**
1. Add a Render Disk in the dashboard
2. Mount it to `/opt/render/project/src/database`
3. Or modify the code to use cloud storage (S3, Cloudinary, etc.)

---

## Monitoring

**View Logs:**
- Go to Render Dashboard → Your Service → Logs
- Real-time logs show embedding progress, queries, and errors

**Check Health:**
- Render provides a public URL
- Hit the `/` endpoint to verify the service is running

---

## Troubleshooting

### Service won't start
- Check environment variables are set correctly
- View logs for Python errors
- Ensure all dependencies in `requirements.txt` are valid

### Embeddings fail
- Verify `GEMINI_API_KEY` is correct
- Check Gemini API quota (15 req/min)
- View logs for API errors

### Qdrant connection fails
- If using cloud, verify `QDRANT_URL` and `QDRANT_API_KEY`
- Check Qdrant Cloud dashboard for cluster status
- For local mode, use `QDRANT_URL=:memory:`

### Cold starts are slow
- This is normal for Render free tier
- Consider upgrading to paid tier for always-on service
- Use Qdrant Cloud to persist data between restarts

---

## Next Steps

1. **Add CORS configuration** in `app.py` for your frontend
2. **Set up monitoring** with Render's built-in tools
3. **Configure custom domain** in Render dashboard
4. **Add authentication** for production use
5. **Implement rate limiting** to prevent abuse

---

## Cost Breakdown

| Service | Free Tier | Paid Tier Starts At |
|---------|-----------|---------------------|
| Render | 750 hrs/month | $7/month |
| Qdrant Cloud | 1GB forever | $25/month (8GB) |
| Google Gemini | 1.5K req/day | Pay-as-you-go |
| Groq | 30 req/min | Free (for now) |

**Total Monthly Cost:** $0 (with free tiers)

---

## Support

For issues or questions:
- Check [Render Docs](https://render.com/docs)
- Check [Qdrant Docs](https://qdrant.tech/documentation/)
- Check [Gemini API Docs](https://ai.google.dev/docs)
- Check [Groq Docs](https://console.groq.com/docs)
