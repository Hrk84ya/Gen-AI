# Recommendation Systems

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand collaborative filtering, content-based, and hybrid approaches
- Implement matrix factorization from scratch
- Build a neural collaborative filtering model
- Understand evaluation metrics for recommender systems (NDCG, Hit Rate, MAP)
- Know how modern production recommenders work (two-tower, retrieval + ranking)

## 🛒 Why Recommendation Systems?

Recommender systems are one of the most commercially impactful applications of ML. They power product suggestions on e-commerce sites, content feeds on social media, movie/music recommendations, and ad targeting. Understanding them bridges ML theory with real business value.

### Approaches
- **Collaborative Filtering**: "Users who liked X also liked Y"
- **Content-Based Filtering**: "This item is similar to items you liked"
- **Hybrid**: Combine both signals
- **Deep Learning**: Neural models that learn complex user-item interactions

## 📚 Module Contents

### 1. [Classical & Neural Recommenders](./01_recommendation_systems.py)
- Collaborative filtering (user-based, item-based)
- Matrix Factorization (SVD-style)
- Neural Collaborative Filtering (NCF)
- Evaluation metrics (NDCG, Hit Rate, Precision@K)

### 2. [Advanced Recommenders](./02_advanced_recommenders.py)
- Two-tower retrieval model
- Sequence-aware recommendations (session-based)
- Hybrid content + collaborative model
- Cold-start strategies

## 📚 Additional Resources

### Papers
- "Matrix Factorization Techniques for Recommender Systems" (Koren et al., 2009)
- "Neural Collaborative Filtering" (He et al., 2017)
- "Deep Learning based Recommender System: A Survey and New Perspectives" (Zhang et al., 2019)

### Libraries
- [Surprise](https://surpriselib.com/) — scikit-learn style recommender library
- [LightFM](https://making.lyst.com/lightfm/docs/) — hybrid recommender
- [RecBole](https://recbole.io/) — unified recommendation framework

### Online
- [Google's Recommendation Systems Course](https://developers.google.com/machine-learning/recommendation)
- [Stanford CS246: Mining Massive Datasets (Recommender chapter)](http://www.mmds.org/)

---
*Back to [Advanced Overview](../../04-Advanced/) →*
