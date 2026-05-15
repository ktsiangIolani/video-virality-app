import { useState } from 'react';
import { Eye, ThumbsUp, Upload, Play, FileText, MessageSquare, Image, Tag, TrendingUp, BarChart2 } from 'lucide-react';

export default function App() {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [thumbnail, setThumbnail] = useState(null);
  const [thumbnailPreview, setThumbnailPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleThumbnailChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setThumbnail(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setThumbnailPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);

    // 1. Pack all the form data into a neat package
    const formData = new FormData();
    formData.append('title', title);
    formData.append('description', description);
    formData.append('tags', tags);
    if (thumbnail) {
      formData.append('thumbnail', thumbnail);
    } else {
      alert("Please upload a thumbnail!");
      setIsLoading(false);
      return;
    }

    try {
      // 2. Send the package to the Python Flask server
      // FIX: route is /api/predict (not /predict) to match app.py's @app.route('/api/predict')
      const response = await fetch(
        "https://video-virality-app-be-669394454391.europe-west1.run.app/api/predict",
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error(`Server responded with a ${response.status} status.`);
      }

      // 3. Receive the prediction and update the UI
      const data = await response.json();

      if (!data.success || data.error) {
        console.error("Backend Error:", data.error);
        alert("There was an error processing your prediction.");
      } else {
        setPrediction({ views: data.views, likes: data.likes });
      }

    } catch (error) {
      console.error("Connection Error:", error);
      alert("Could not connect to the backend server. Make sure app.py is running!");
    } finally {
      setIsLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  // Derived metrics computed from the views + likes the backend returns
  const getRevenue = (views) => {
    const revenue = views * 0.001;
    if (revenue >= 1000) return '$' + (revenue / 1000).toFixed(1) + 'K';
    return '$' + revenue.toFixed(2);
  };

  const getViralityScore = (views) => {
    // Logarithmic scale: 10M+ views = 10/10
    const score = Math.min(10, Math.round((Math.log10(Math.max(views, 1)) / Math.log10(10_000_000)) * 10));
    if (score >= 9) return { score, label: '🔥 Mega Viral' };
    if (score >= 7) return { score, label: '🚀 Viral' };
    if (score >= 5) return { score, label: '📈 Trending' };
    if (score >= 3) return { score, label: '👍 Solid' };
    return { score, label: '🌱 Growing' };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white via-red-50 to-white">
      {/* YouTube-themed Header Bar */}
      <div className="bg-white border-b-2 border-red-600 shadow-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="bg-red-600 rounded-lg p-2 shadow-lg">
                <Play className="w-8 h-8 text-white fill-white" />
              </div>
              <div className="flex flex-col">
                <span className="text-2xl tracking-tight" style={{ fontFamily: 'Arial, sans-serif' }}>
                  <span className="text-gray-900">Virality</span>
                  <span className="text-red-600">Predictor</span>
                </span>
                <span className="text-xs text-gray-500 -mt-1">Powered by YouTube Analytics</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-block bg-red-600 text-white px-6 py-3 rounded-full mb-4 shadow-lg">
            <span className="text-lg">🎬 AI-Powered Predictions</span>
          </div>
          <h1 className="text-5xl mb-4 text-gray-900">
            Will Your Video Go Viral?
          </h1>
          <p className="text-xl text-gray-600">
            Get instant predictions on views and likes before you upload! 🚀
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-white border-2 border-red-100 rounded-2xl p-8 shadow-xl mb-8">
          {/* Title Input */}
          <div className="mb-6">
            <label htmlFor="title" className="block mb-3 text-gray-800 flex items-center gap-3 text-xl">
              <FileText className="w-6 h-6 text-red-600" /> Video Title
            </label>
            <input
              id="title"
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter your video title..."
              required
              className="w-full px-4 py-3 border-2 border-red-200 rounded-lg focus:border-red-600 focus:outline-none transition-colors bg-white hover:border-red-300"
            />
          </div>

          {/* Description Input */}
          <div className="mb-6">
            <label htmlFor="description" className="block mb-3 text-gray-800 flex items-center gap-3 text-xl">
              <MessageSquare className="w-6 h-6 text-red-600" /> Video Description
            </label>
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter your video description..."
              required
              rows={4}
              className="w-full px-4 py-3 border-2 border-red-200 rounded-lg focus:border-red-600 focus:outline-none transition-colors resize-none bg-white hover:border-red-300"
            />
          </div>

          {/* Tags Input */}
          <div className="mb-6">
            <label htmlFor="tags" className="block mb-3 text-gray-800 flex items-center gap-3 text-xl">
              <Tag className="w-6 h-6 text-red-600" /> Video Tags
            </label>
            <input
              id="tags"
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="Enter video tags separated by commas..."
              required
              className="w-full px-4 py-3 border-2 border-red-200 rounded-lg focus:border-red-600 focus:outline-none transition-colors bg-white hover:border-red-300"
            />
          </div>

          {/* Thumbnail Upload */}
          <div className="mb-8">
            <label className="block mb-3 text-gray-800 flex items-center gap-3 text-xl">
              <Image className="w-6 h-6 text-red-600" /> Video Thumbnail
            </label>
            <div className="relative">
              <input
                id="thumbnail"
                type="file"
                accept="image/*"
                onChange={handleThumbnailChange}
                required
                className="hidden"
              />
              <label
                htmlFor="thumbnail"
                className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-red-300 rounded-lg cursor-pointer hover:border-red-600 transition-all bg-red-50/30 hover:bg-red-600 group"
              >
                {thumbnailPreview ? (
                  <div className="relative w-full h-full">
                    <img
                      src={thumbnailPreview}
                      alt="Thumbnail preview"
                      className="w-full h-full object-cover rounded-lg"
                    />
                    <div className="absolute top-2 right-2 bg-red-600 text-white px-3 py-1 rounded text-xs shadow-lg">
                      Preview
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <div className="bg-red-100 group-hover:bg-white/20 p-4 rounded-full mb-3 transition-colors">
                      <Upload className="w-8 h-8 text-red-600 group-hover:text-white transition-colors" />
                    </div>
                    <span className="text-gray-700 group-hover:text-white transition-colors">Click to upload thumbnail</span>
                    <span className="text-sm text-gray-500 group-hover:text-red-100 mt-1 transition-colors">PNG, JPG up to 10MB</span>
                  </div>
                )}
              </label>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-4 px-6 bg-red-600 text-white border-2 border-red-600 rounded-full transition-all duration-300 ease-in-out hover:bg-red-700 hover:border-red-700 hover:scale-105 active:scale-100 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:bg-red-600 disabled:hover:border-red-600 shadow-lg hover:shadow-xl"
          >
            <span className="text-lg">
              {isLoading ? '⏳ Analyzing Your Content...' : '🎯 Predict My Virality!'}
            </span>
          </button>
        </form>

        {/* Results */}
        {prediction && (
          <div className="bg-white border-2 border-red-600 rounded-2xl p-8 shadow-2xl animate-fadeIn">
            <div className="text-center mb-8">
              <div className="inline-block bg-gradient-to-r from-red-600 to-red-500 text-white px-8 py-3 rounded-full shadow-lg mb-4">
                <h2 className="text-2xl">🎉 Your Prediction Results</h2>
              </div>
              <p className="text-gray-600">Based on advanced AI analysis</p>
            </div>

            {/* Row 1: Views + Likes (from backend) */}
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              {/* Views */}
              <div className="bg-gradient-to-br from-red-50 to-white rounded-xl p-8 border-2 border-red-200 shadow-lg hover:shadow-xl transition-all hover:scale-105">
                <div className="flex items-center justify-center mb-4">
                  <div className="w-20 h-20 bg-gradient-to-br from-red-500 to-red-600 rounded-full flex items-center justify-center shadow-lg">
                    <Eye className="w-10 h-10 text-white" />
                  </div>
                </div>
                <div className="text-center">
                  <p className="text-gray-600 mb-2 text-lg">Predicted Views</p>
                  <p className="text-5xl text-red-600 mb-1" style={{ fontFamily: 'Arial, sans-serif' }}>
                    {formatNumber(prediction.views)}
                  </p>
                  <div className="inline-block bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm mt-2">
                    👀 Visibility Score
                  </div>
                </div>
              </div>

              {/* Likes */}
              <div className="bg-gradient-to-br from-red-50 to-white rounded-xl p-8 border-2 border-red-200 shadow-lg hover:shadow-xl transition-all hover:scale-105">
                <div className="flex items-center justify-center mb-4">
                  <div className="w-20 h-20 bg-gradient-to-br from-red-500 to-red-600 rounded-full flex items-center justify-center shadow-lg">
                    <ThumbsUp className="w-10 h-10 text-white" />
                  </div>
                </div>
                <div className="text-center">
                  <p className="text-gray-600 mb-2 text-lg">Predicted Likes</p>
                  <p className="text-5xl text-red-600 mb-1" style={{ fontFamily: 'Arial, sans-serif' }}>
                    {formatNumber(prediction.likes)}
                  </p>
                  <div className="inline-block bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm mt-2">
                    👍 Engagement Score
                  </div>
                </div>
              </div>
            </div>

            {/* Row 2: Derived metrics (computed from views + likes) */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Predicted Revenue */}
              <div className="bg-gradient-to-br from-red-50 to-white rounded-xl p-8 border-2 border-red-200 shadow-lg hover:shadow-xl transition-all hover:scale-105">
                <div className="flex items-center justify-center mb-4">
                  <div className="w-20 h-20 bg-gradient-to-br from-red-400 to-red-500 rounded-full flex items-center justify-center shadow-lg">
                    <BarChart2 className="w-10 h-10 text-white" />
                  </div>
                </div>
                <div className="text-center">
                  <p className="text-gray-600 mb-2 text-lg">Predicted Revenue</p>
                  <p className="text-5xl text-red-600 mb-1" style={{ fontFamily: 'Arial, sans-serif' }}>
                    {getRevenue(prediction.views)}
                  </p>
                  <div className="inline-block bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm mt-2">
                    💰 Views × $0.001
                  </div>
                </div>
              </div>

              {/* Virality Score */}
              <div className="bg-gradient-to-br from-red-50 to-white rounded-xl p-8 border-2 border-red-200 shadow-lg hover:shadow-xl transition-all hover:scale-105">
                <div className="flex items-center justify-center mb-4">
                  <div className="w-20 h-20 bg-gradient-to-br from-red-400 to-red-500 rounded-full flex items-center justify-center shadow-lg">
                    <TrendingUp className="w-10 h-10 text-white" />
                  </div>
                </div>
                <div className="text-center">
                  <p className="text-gray-600 mb-2 text-lg">Virality Score</p>
                  <p className="text-5xl text-red-600 mb-1" style={{ fontFamily: 'Arial, sans-serif' }}>
                    {getViralityScore(prediction.views).score}
                    <span className="text-2xl text-gray-400"> / 10</span>
                  </p>
                  <p className="text-lg font-semibold text-red-500 mb-1">
                    {getViralityScore(prediction.views).label}
                  </p>
                  <div className="inline-block bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm mt-1">
                    🏆 Overall Potential
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}