# Website Analytics Setup Guide

This guide will help you set up visitor tracking for your GitHub Pages website using Google Analytics 4.

## 🚀 Quick Setup (5 minutes)

### Step 1: Create Google Analytics Account

1. Go to [Google Analytics](https://analytics.google.com/)
2. Click "Start measuring" or "Get started"
3. Create an account or use existing Google account
4. Set up a property for your website:
   - Property name: "Junlian's Tech Blog" (or your preferred name)
   - Reporting time zone: Select your timezone
   - Currency: Select your currency

### Step 2: Create Data Stream

1. In your GA4 property, go to **Admin** → **Data Streams**
2. Click **Add stream** → **Web**
3. Enter your website details:
   - Website URL: `https://junlian.github.io`
   - Stream name: "Main Website"
4. Click **Create stream**

### Step 3: Get Your Measurement ID

1. After creating the stream, you'll see your **Measurement ID** (format: `G-XXXXXXXXXX`)
2. Copy this ID

### Step 4: Update Your Website Configuration

1. Open `_config.yml` in your repository
2. Replace `G-XXXXXXXXXX` with your actual Measurement ID:
   ```yaml
   google_analytics: G-YOUR-ACTUAL-ID
   ```
3. Save and commit the changes

### Step 5: Deploy and Test

1. Push your changes to GitHub
2. Wait 5-10 minutes for GitHub Pages to rebuild
3. Visit your website and check if tracking is working

## 📊 What You'll Track

Once set up, you'll have access to:

- **Real-time visitors** - See who's on your site right now
- **Page views** - Most popular blog posts and pages
- **Traffic sources** - Where visitors come from (Google, social media, direct)
- **Geographic data** - Countries and cities of your visitors
- **Device info** - Desktop vs mobile usage
- **User behavior** - How long people stay, bounce rate
- **Conversion tracking** - Track specific goals (newsletter signups, etc.)

## 🔒 Privacy Features Included

Your setup includes privacy-friendly features:
- **IP Anonymization** - Visitor IPs are anonymized
- **Cookie compliance** - Secure cookie settings
- **GDPR friendly** - Respects user privacy preferences

## 🎯 Alternative Analytics Options

If you prefer privacy-focused alternatives, you can also use:

### Plausible Analytics (Recommended Alternative)
- Privacy-friendly, no cookies
- GDPR compliant by default
- Lightweight (< 1KB script)
- Simple, clean interface

To use Plausible:
1. Sign up at [plausible.io](https://plausible.io)
2. Add your domain: `junlian.github.io`
3. Uncomment the Plausible line in `_includes/head.html`
4. Update `_config.yml` with: `plausible_domain: junlian.github.io`

### Simple Analytics
- Similar to Plausible
- Privacy-focused
- No cookies, GDPR compliant

## 🔧 Troubleshooting

### Analytics Not Working?
1. **Check Measurement ID** - Ensure it's correct in `_config.yml`
2. **Wait for deployment** - GitHub Pages takes 5-10 minutes to update
3. **Clear browser cache** - Hard refresh your website
4. **Check Real-Time reports** - Visit your site and check GA4 Real-Time view

### Testing Your Setup
1. Visit your website
2. Go to Google Analytics → Reports → Real-time
3. You should see your visit appear within 30 seconds

## 📈 Next Steps

Once analytics is working:
1. **Set up goals** - Track newsletter signups, course enrollments
2. **Create custom events** - Track blog post engagement
3. **Set up alerts** - Get notified of traffic spikes
4. **Connect Google Search Console** - See which keywords bring traffic

## 🎓 Learning Resources

- [Google Analytics Academy](https://analytics.google.com/analytics/academy/)
- [GA4 Setup Guide](https://support.google.com/analytics/answer/9304153)
- [Jekyll Analytics Documentation](https://jekyllrb.com/docs/usage/)

---

**Need help?** The analytics code is already integrated into your website. Just replace the placeholder ID with your actual Google Analytics Measurement ID and you're ready to track visitors!