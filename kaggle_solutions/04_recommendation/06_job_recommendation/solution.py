"""
Job Recommendation System using Hybrid Filtering
=================================================
This solution demonstrates a job recommendation system combining
content-based filtering (skills matching) and collaborative filtering
(similar candidate preferences).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class JobRecommender:
    """Hybrid job recommendation system."""

    def __init__(self, content_weight=0.6, collab_weight=0.4):
        """
        Initialize recommender.

        Args:
            content_weight: Weight for content-based score
            collab_weight: Weight for collaborative score
        """
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.job_skill_matrix = None
        self.candidate_skill_matrix = None
        self.user_item_matrix = None

    def generate_data(self, n_jobs=400, n_candidates=250):
        """
        Generate synthetic job and candidate data.

        Args:
            n_jobs: Number of job postings
            n_candidates: Number of candidates

        Returns:
            jobs_df, candidates_df, applications_df
        """
        print("Generating synthetic job data...")

        # Define skill taxonomy
        all_skills = {
            'programming': ['Python', 'Java', 'JavaScript', 'C++', 'R', 'SQL', 'Go', 'Ruby'],
            'data': ['Machine Learning', 'Deep Learning', 'Data Analysis', 'Statistics', 'Big Data', 'Spark'],
            'web': ['React', 'Angular', 'Node.js', 'Django', 'Flask', 'HTML/CSS', 'REST API'],
            'cloud': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD'],
            'database': ['PostgreSQL', 'MongoDB', 'Redis', 'MySQL', 'Cassandra'],
            'other': ['Git', 'Agile', 'Leadership', 'Communication', 'Problem Solving']
        }

        flat_skills = [skill for skills in all_skills.values() for skill in skills]

        # Job titles and their typical skill requirements
        job_templates = {
            'Data Scientist': ['Python', 'Machine Learning', 'Statistics', 'SQL', 'Deep Learning'],
            'Software Engineer': ['Python', 'Java', 'Git', 'Problem Solving', 'Agile'],
            'Web Developer': ['JavaScript', 'React', 'Node.js', 'HTML/CSS', 'REST API'],
            'ML Engineer': ['Python', 'Machine Learning', 'Deep Learning', 'AWS', 'Docker'],
            'Backend Engineer': ['Python', 'Java', 'SQL', 'REST API', 'PostgreSQL'],
            'DevOps Engineer': ['AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Python'],
            'Data Engineer': ['Python', 'SQL', 'Spark', 'Big Data', 'AWS'],
            'Full Stack Developer': ['JavaScript', 'React', 'Node.js', 'Python', 'PostgreSQL']
        }

        locations = ['New York', 'San Francisco', 'Austin', 'Seattle', 'Boston', 'Remote']
        experience_levels = ['Entry', 'Mid', 'Senior', 'Lead']
        companies = [f'Company_{i}' for i in range(1, 101)]

        # Generate jobs
        jobs = []
        for job_id in range(n_jobs):
            title = np.random.choice(list(job_templates.keys()))
            base_skills = job_templates[title].copy()

            # Add some variation
            n_extra = np.random.randint(0, 4)
            extra_skills = np.random.choice([s for s in flat_skills if s not in base_skills],
                                          size=min(n_extra, len(flat_skills) - len(base_skills)),
                                          replace=False)
            required_skills = base_skills + list(extra_skills)

            jobs.append({
                'job_id': job_id,
                'title': title,
                'company': np.random.choice(companies),
                'location': np.random.choice(locations),
                'experience_level': np.random.choice(experience_levels),
                'required_skills': required_skills,
                'salary': np.random.randint(60, 200) * 1000
            })

        jobs_df = pd.DataFrame(jobs)

        # Generate candidates
        candidates = []
        for candidate_id in range(n_candidates):
            # Candidates specialize in certain areas
            specialization = np.random.choice(list(all_skills.keys()))
            base_skills = list(np.random.choice(all_skills[specialization],
                                               size=min(4, len(all_skills[specialization])),
                                               replace=False))

            # Add skills from other areas
            other_skills = [s for cat, skills in all_skills.items() if cat != specialization for s in skills]
            n_other = np.random.randint(2, 6)
            base_skills.extend(list(np.random.choice(other_skills, size=min(n_other, len(other_skills)), replace=False)))

            candidates.append({
                'candidate_id': candidate_id,
                'skills': base_skills,
                'experience_years': np.random.randint(0, 15),
                'preferred_location': np.random.choice(locations),
                'min_salary': np.random.randint(50, 150) * 1000
            })

        candidates_df = pd.DataFrame(candidates)

        # Generate applications (candidates apply to jobs)
        applications = []
        for candidate_id in range(n_candidates):
            candidate = candidates_df.iloc[candidate_id]
            candidate_skills = set(candidate['skills'])

            # Candidates apply to 5-20 jobs
            n_applications = np.random.randint(5, 21)

            for _ in range(n_applications):
                job = jobs_df.sample(1).iloc[0]
                job_skills = set(job['required_skills'])

                # Calculate skill match
                skill_match = len(candidate_skills & job_skills) / len(job_skills) if len(job_skills) > 0 else 0

                # Higher match means more likely to be interested
                if skill_match > 0.4:
                    interest_score = np.random.uniform(0.6, 1.0)
                else:
                    interest_score = np.random.uniform(0.1, 0.5)

                # Application outcome
                if skill_match > 0.5 and interest_score > 0.7:
                    status = np.random.choice(['Applied', 'Interviewed', 'Offered'], p=[0.5, 0.3, 0.2])
                else:
                    status = 'Applied'

                applications.append({
                    'candidate_id': candidate_id,
                    'job_id': job['job_id'],
                    'interest_score': interest_score,
                    'skill_match': skill_match,
                    'status': status
                })

        applications_df = pd.DataFrame(applications)
        applications_df = applications_df.drop_duplicates(['candidate_id', 'job_id'])

        print(f"Generated {len(jobs_df)} jobs, {len(candidates_df)} candidates, {len(applications_df)} applications")

        self.jobs_df = jobs_df
        self.candidates_df = candidates_df
        self.all_skills = flat_skills

        return jobs_df, candidates_df, applications_df

    def build_content_model(self):
        """Build content-based model using skill matching."""
        print("\nBuilding content-based model...")

        # Create skill binary matrix for jobs
        mlb_jobs = MultiLabelBinarizer()
        self.job_skill_matrix = mlb_jobs.fit_transform(self.jobs_df['required_skills'])
        self.job_skills_labels = mlb_jobs.classes_

        # Create skill binary matrix for candidates
        mlb_candidates = MultiLabelBinarizer()
        mlb_candidates.fit([self.job_skills_labels])  # Use same features
        self.candidate_skill_matrix = mlb_candidates.transform(self.candidates_df['skills'])

        print(f"Skill feature dimension: {len(self.job_skills_labels)}")

    def build_collaborative_model(self, applications_df):
        """Build collaborative filtering model."""
        print("Building collaborative model...")

        # Create candidate-job interaction matrix
        interaction_matrix = applications_df.pivot_table(
            index='candidate_id',
            columns='job_id',
            values='interest_score',
            fill_value=0
        )

        self.user_item_matrix = interaction_matrix
        print(f"Interaction matrix shape: {self.user_item_matrix.shape}")

    def content_score(self, candidate_id, job_id):
        """Calculate content-based score."""
        if candidate_id >= len(self.candidate_skill_matrix) or job_id >= len(self.job_skill_matrix):
            return 0.0

        candidate_vector = self.candidate_skill_matrix[candidate_id]
        job_vector = self.job_skill_matrix[job_id]

        # Cosine similarity
        similarity = cosine_similarity([candidate_vector], [job_vector])[0][0]
        return similarity

    def collaborative_score(self, candidate_id, job_id):
        """Calculate collaborative filtering score."""
        if candidate_id not in self.user_item_matrix.index or job_id not in self.user_item_matrix.columns:
            return 0.0

        # Find similar candidates
        candidate_vector = self.user_item_matrix.loc[candidate_id].values
        all_candidates = self.user_item_matrix.values

        similarities = cosine_similarity([candidate_vector], all_candidates)[0]

        # Top 10 similar candidates
        top_k = 10
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]

        # Weighted average of their scores
        scores = []
        weights = []
        for idx in similar_indices:
            sim_candidate_id = self.user_item_matrix.index[idx]
            if job_id in self.user_item_matrix.columns:
                score = self.user_item_matrix.loc[sim_candidate_id, job_id]
                if score > 0:
                    scores.append(score)
                    weights.append(similarities[idx])

        if len(scores) > 0:
            return np.average(scores, weights=weights)
        return 0.0

    def hybrid_recommend(self, candidate_id, n=10, exclude_applied=True, applications_df=None):
        """
        Generate hybrid recommendations.

        Args:
            candidate_id: Candidate ID
            n: Number of recommendations
            exclude_applied: Exclude already applied jobs
            applications_df: Application history

        Returns:
            List of (job_id, score) tuples
        """
        if candidate_id >= len(self.candidates_df):
            return []

        scores = []
        for job_id in range(len(self.jobs_df)):
            content = self.content_score(candidate_id, job_id)
            collab = self.collaborative_score(candidate_id, job_id)

            # Hybrid score
            hybrid_score = self.content_weight * content + self.collab_weight * collab
            scores.append((job_id, hybrid_score, content, collab))

        # Exclude applied jobs
        if exclude_applied and applications_df is not None:
            applied_jobs = set(applications_df[applications_df['candidate_id'] == candidate_id]['job_id'].values)
            scores = [(jid, score, c, co) for jid, score, c, co in scores if jid not in applied_jobs]

        # Sort by hybrid score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:n]

    def evaluate(self, applications_df, test_applications_df):
        """Evaluate recommendation quality."""
        print("\nEvaluating model...")

        precision_scores = []
        recall_scores = []
        ndcg_scores = []

        k = 10

        for candidate_id in test_applications_df['candidate_id'].unique():
            if candidate_id >= len(self.candidates_df):
                continue

            # Get recommendations
            recs = self.hybrid_recommend(candidate_id, n=k, exclude_applied=True, applications_df=applications_df)
            rec_jobs = [job_id for job_id, _, _, _ in recs]

            # Get actual interested jobs (interest_score > 0.6)
            actual_jobs = test_applications_df[
                (test_applications_df['candidate_id'] == candidate_id) &
                (test_applications_df['interest_score'] > 0.6)
            ]['job_id'].values

            if len(actual_jobs) > 0:
                hits = len(set(rec_jobs) & set(actual_jobs))
                precision_scores.append(hits / k if k > 0 else 0)
                recall_scores.append(hits / len(actual_jobs))

                # NDCG
                dcg = sum([1.0 / np.log2(i + 2) for i, jid in enumerate(rec_jobs) if jid in actual_jobs])
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_jobs), k))])
                ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

        metrics = {
            'precision@10': np.mean(precision_scores) if precision_scores else 0,
            'recall@10': np.mean(recall_scores) if recall_scores else 0,
            'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0
        }

        print(f"Precision@10: {metrics['precision@10']:.4f}")
        print(f"Recall@10: {metrics['recall@10']:.4f}")
        print(f"NDCG@10: {metrics['ndcg@10']:.4f}")

        return metrics

    def visualize_results(self, applications_df):
        """Create visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Job title distribution
        ax = axes[0, 0]
        title_counts = self.jobs_df['title'].value_counts()
        title_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Job Title')
        ax.set_ylabel('Count')
        ax.set_title('Job Postings by Title')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 2. Skill match distribution
        ax = axes[0, 1]
        ax.hist(applications_df['skill_match'], bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Skill Match Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Skill Match Distribution')
        ax.grid(True, alpha=0.3)

        # 3. Interest score distribution
        ax = axes[1, 0]
        ax.hist(applications_df['interest_score'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Interest Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Candidate Interest Distribution')
        ax.grid(True, alpha=0.3)

        # 4. Applications per candidate
        ax = axes[1, 1]
        app_counts = applications_df['candidate_id'].value_counts()
        ax.hist(app_counts, bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Number of Applications')
        ax.set_ylabel('Number of Candidates')
        ax.set_title('Application Activity Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/job_recommendation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to /tmp/job_recommendation_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Job Recommendation System using Hybrid Filtering")
    print("=" * 70)

    # Initialize recommender
    recommender = JobRecommender(content_weight=0.6, collab_weight=0.4)

    # Generate data
    jobs_df, candidates_df, applications_df = recommender.generate_data(n_jobs=400, n_candidates=250)

    # Split applications
    train_applications = applications_df.sample(frac=0.8, random_state=42)
    test_applications = applications_df.drop(train_applications.index)

    print(f"\nTrain applications: {len(train_applications)}")
    print(f"Test applications: {len(test_applications)}")

    # Build models
    recommender.build_content_model()
    recommender.build_collaborative_model(train_applications)

    # Evaluate
    metrics = recommender.evaluate(train_applications, test_applications)

    # Example recommendations
    print("\n" + "=" * 70)
    print("Example Recommendations")
    print("=" * 70)

    test_candidate = train_applications['candidate_id'].value_counts().head(1).index[0]
    candidate = candidates_df.iloc[test_candidate]

    print(f"\nRecommendations for Candidate {test_candidate}:")
    print(f"Skills: {', '.join(candidate['skills'][:5])}...")
    print(f"Experience: {candidate['experience_years']} years")
    print(f"Preferred location: {candidate['preferred_location']}")

    recommendations = recommender.hybrid_recommend(test_candidate, n=10, exclude_applied=True, applications_df=train_applications)
    print("\nTop 10 Recommended Jobs:")
    for i, (job_id, hybrid_score, content_score, collab_score) in enumerate(recommendations, 1):
        job = jobs_df.iloc[job_id]
        print(f"{i}. {job['title']} at {job['company']} ({job['location']})")
        print(f"   Hybrid: {hybrid_score:.3f} | Content: {content_score:.3f} | Collab: {collab_score:.3f}")

    # Visualize
    recommender.visualize_results(applications_df)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
