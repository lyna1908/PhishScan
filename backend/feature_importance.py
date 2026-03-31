import matplotlib.pyplot as plt
import joblib

model = joblib.load('model/best_model.pkl')

features = ['url_count', 'has_ip_url', 'has_short_url',
            'urgent_keyword', 'is_free_email', 'subject_urgent',
            'body_length', 'has_html', 'html_text_ratio', 'urls']

importance = model.feature_importances_

# Sort by importance
sorted_pairs = sorted(zip(importance, features), reverse=True)
sorted_importance, sorted_features = zip(*sorted_pairs)

plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
bars = plt.barh(sorted_features, sorted_importance, color='#00ff41')
plt.xlabel('Importance Score', color='#00ff41')
plt.title('Feature Importance — Random Forest', color='#00ff41')
plt.tick_params(colors='#00ff41')
plt.gca().spines['bottom'].set_color('#00ff41')
plt.gca().spines['left'].set_color('#00ff41')
plt.tight_layout()
plt.savefig('../results/feature_importance.png',
            facecolor='#0a0a0a', edgecolor='none')
print("✅ feature_importance.png saved!")
