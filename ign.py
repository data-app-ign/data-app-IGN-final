import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import (mean_squared_error, accuracy_score,
                             classification_report, roc_auc_score, roc_curve, mean_absolute_error, r2_score)
from sklearn.metrics.pairwise import cosine_similarity

uploaded_file = None

st.sidebar.title("Gaming Industry Analytics Suite")
section = st.sidebar.radio("Go to", [
    "► 1. Overview and Objectives",
    "► 2. Data integration and overview",
    "► 3. Game Success Prediction Analysis",
    "► 4. Post-Release Market Impact Analysis",
    "► 5. Enhancing User Engagement and Sales Strategies"
])

if section == "► 1. Overview and Objectives":
    st.title("Comprehensive Analysis Tool for the Gaming Industry")
    st.header("1.1. Objectives and Target Audience")
    st.write(
        """
        The gaming industry is rapidly growing and has already reached a total market capitalization of several hundred billion USD. This has made success a lucrative prospect, but, as in any lucrative industry, competition is fierce, with many amazing games being released each year. It is becoming increasingly difficult to stand apart, reflected by the ballooning average marketing costs associated with modern game releases.
        This project aims to bring a quantitative approach, driven by real world data, to better help game developers schedule their game releases to maximize their success potential and increase their likelihood of achieving critical acclaim. Predictive insights to recommend those annual release date windows most conducive to success are implemented.
        """
    )

    st.header("1.2. Application and Benefits")
    st.write(
        """
        The core value of this project's predictive insights is based on the significant statistical impact a game's release date has on the average critical acclaim any given game across genres has. Beyond proving statistical significance, this project recommends game developers release windows most conducive to their success, and demonstrates the predicted impact "gaming" the release date has on potential success parameters, primarily that of critical acclaim.
        """
    )

    st.header("1.3. Key Features and Real-world Application")
    st.write(
        """
        Features include trend analysis, success prediction algorithms, 
        and interactive visualizations. Real-world applications range from selecting optimal game release dates 
        to tailoring marketing strategies based on consumer preferences.
        """
    )


if section == "► 2. Data integration and overview":
    st.title("Load the dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8").strip()
        if not file_content or all(line.isspace() for line in file_content.splitlines()):
            st.warning("Uploaded file is empty or contains only whitespace. Please upload a valid CSV file.")
        else:
            try:
                from io import StringIO

                st.session_state.data = pd.read_csv(StringIO(file_content))
                st.success("Dataset successfully uploaded!")
            except pd.errors.EmptyDataError:
                st.error("The uploaded file contains no data or columns. Please upload a valid CSV file.")
            except Exception as e:
                st.error(f"Error loading the file: {e}")

    if 'data' in st.session_state:
        st.session_state.data['score'] = st.session_state.data['score'].astype('float64')
        st.session_state.data['release_date'] = pd.to_datetime(
            st.session_state.data['release_year'].astype(str) + '-' +
            st.session_state.data['release_month'].astype(
                str) + '-' +
            st.session_state.data['release_day'].astype(str))

        duplicated_urls = st.session_state.data[st.session_state.data['url'].duplicated(keep=False)].sort_values(
            by='url')
        st.session_state.data = st.session_state.data.drop_duplicates(subset='url', keep='first')
        st.session_state.data.reset_index(drop=True, inplace=True)
        st.session_state.data['release_date'] = pd.to_datetime(st.session_state.data['release_year'].astype(str) + '-' +
                                                               st.session_state.data['release_month'].astype(
                                                                   str) + '-' +
                                                               st.session_state.data['release_day'].astype(str))

        st.header("2.1. Explore Games")
        platforms = st.session_state.data['platform'].unique()
        selected_platform = st.selectbox('Select a Platform (Optional)', ['All'] + list(platforms))
        genres = st.session_state.data['genre'].unique()
        selected_genre = st.selectbox('Select a Genre (Optional)', ['All'] + list(genres))
        year_to_filter = st.slider('Select a Year',
                                   min_value=int(st.session_state.data['release_year'].min()),
                                   max_value=int(st.session_state.data['release_year'].max()),
                                   value=int(st.session_state.data['release_year'].min()))
        filtered_data = st.session_state.data[st.session_state.data['release_year'] == year_to_filter]

        if selected_platform != 'All':
            filtered_data = filtered_data[filtered_data['platform'] == selected_platform]

        if selected_genre != 'All':
            filtered_data = filtered_data[filtered_data['genre'] == selected_genre]

        filtered_data = filtered_data.sort_values(by='score', ascending=False)
        if not filtered_data.empty:
            st.write(f"Games released in {year_to_filter}:")
            st.dataframe(filtered_data[['title', 'platform', 'genre', 'score', 'editors_choice']], width=5000)
        else:
            st.write("No games found for the selected filters.")

        st.header("2.2. Game Ratings Analysis")
        analysis_type = st.radio("Choose the Analysis Type",
                                 ('Ratings Distribution', 'Top Rated Games', 'Ratings Over Time',
                                  'Platform-Wise Rating Comparison'))

        if analysis_type == 'Ratings Distribution':
            fig, ax = plt.subplots()
            ax.hist(st.session_state.data['score'], bins=30, edgecolor='black')
            ax.set_title('Distribution of Game Scores')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        elif analysis_type == 'Top Rated Games':
            top_n = st.slider('Select number of top rated games to display', 5, 50, 10)
            top_rated_games = st.session_state.data.sort_values(by='score', ascending=False).head(top_n)
            st.dataframe(top_rated_games[['title', 'platform', 'score']])

        elif analysis_type == 'Ratings Over Time':
            ratings_over_time = st.session_state.data.groupby('release_year')['score'].mean().reset_index()
            st.line_chart(ratings_over_time, x='release_year', y='score')

        elif analysis_type == 'Platform-Wise Rating Comparison':
            platform_ratings = st.session_state.data.groupby('platform')['score'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 15))
            ax.barh(platform_ratings['platform'], platform_ratings['score'])
            ax.set_title('Average Ratings per Platform')
            ax.set_xlabel('Average Score')
            ax.set_ylabel('Platform')
            plt.tight_layout()
            st.pyplot(fig)

        st.header("2.3. Game Finder")

        search_title = st.text_input('Search game by title')
        if search_title:
            results_title = st.session_state.data[st.session_state.data['title'].str.contains(search_title, case=False)]
            if not results_title.empty:
                st.dataframe(results_title[['title', 'platform', 'score', 'release_date']])
            else:
                st.write("No games found with that title.")

        min_score, max_score = st.slider("Select a score range", 0.0, 10.0, (0.0, 10.0))
        results_score = st.session_state.data[
            (st.session_state.data['score'] >= min_score) & (st.session_state.data['score'] <= max_score)]
        if not results_score.empty:
            st.dataframe(results_score[['title', 'platform', 'score', 'release_date']])
        else:
            st.write("No games found in that score range.")

        search_keyword = st.text_input('Enter a keyword to search in reviews (URL field)')
        if search_keyword:
            results_keyword = st.session_state.data[st.session_state.data['url'].str.contains(search_keyword, case=False)]
            if not results_keyword.empty:
                st.dataframe(results_keyword[['title', 'platform', 'score', 'release_date']])
            else:
                st.write("No games found with that keyword in reviews.")

elif section == "► 3. Game Success Prediction Analysis":
    st.title("3. Game Success Prediction Analysis")
    st.subheader("3.1. Predicting the success of a game based on its genre, platform, and year of release")

    if 'data' in st.session_state:
        data_for_model = pd.get_dummies(st.session_state.data[['genre', 'platform', 'release_year', 'score']],
                                        drop_first=True)

        y = data_for_model['score']
        X = data_for_model.drop('score', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_choice = st.selectbox("Choose a model to train", ("Decision Tree", "Random Forest", "Gradient Boosting"))

        if model_choice == "Decision Tree":
            max_depth = st.slider("Max Depth for Decision Tree", 1, 20, 5)
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        elif model_choice == "Random Forest":
            n_estimators = st.slider("Number of Estimators for Random Forest", 10, 300, 100)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=5, random_state=1)
        elif model_choice == "Gradient Boosting":
            learning_rate = st.slider("Learning Rate for Gradient Boosting", 0.01, 0.3, 0.1)
            model = XGBRegressor(learning_rate=learning_rate, n_estimators=100, random_state=1)

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"R^2: {r2:.2f}")

        st.markdown("### Make Individual Predictions")
        input_genre = st.selectbox("Select Genre", [''] + list(st.session_state.data['genre'].unique()))
        input_platform = st.selectbox("Select Platform", [''] + list(st.session_state.data['platform'].unique()))
        input_year = st.number_input("Select Release Year", int(st.session_state.data['release_year'].min()),
                                     int(st.session_state.data['release_year'].max()),
                                     int(st.session_state.data['release_year'].min()))

        if input_genre and input_platform and input_year:
            input_df = pd.DataFrame(
                {'genre': [input_genre], 'platform': [input_platform], 'release_year': [input_year]})
            input_dummies = pd.get_dummies(input_df)
            input_data_aligned = pd.concat([pd.DataFrame(columns=X.columns), input_dummies]).fillna(0.0).iloc[-1:]
            prediction = model.predict(scaler.transform(input_data_aligned))
            st.write(f"Predicted Score: {prediction[0]:.2f}")

        if model_choice in ["Decision Tree", "Random Forest"] and hasattr(model, 'feature_importances_'):
            st.markdown("### Feature Importances")
            feature_importance = pd.Series(model.feature_importances_, index=X.columns)
            st.bar_chart(feature_importance.sort_values(ascending=False).head(10))

    st.subheader("3.2. The influence of the time of year on the success of the game")
    st.markdown("### Convert the data")

    if 'data' in st.session_state:
        data_task_4 = st.session_state.data.copy()

        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'

        data_task_4['season'] = data_task_4['release_month'].apply(get_season)
        season_distribution = data_task_4['season'].value_counts()
        st.write(season_distribution)
        selected_season = st.selectbox("Select a Season to Explore", ['Winter', 'Spring', 'Summer', 'Autumn'])
        st.write(f"Games released in {selected_season}:")
        st.dataframe(data_task_4[data_task_4['season'] == selected_season])

        genre_filter = st.multiselect("Filter by Genre", options=data_task_4['genre'].unique())
        platform_filter = st.multiselect("Filter by Platform", options=data_task_4['platform'].unique())
        if genre_filter:
            data_task_4 = data_task_4[data_task_4['genre'].isin(genre_filter)]
        if platform_filter:
            data_task_4 = data_task_4[data_task_4['platform'].isin(platform_filter)]
        st.markdown("### Average Game Rating by Season (Filtered)")
        average_ratings_by_season = data_task_4.groupby('season')['score'].mean().sort_values()
        st.bar_chart(average_ratings_by_season)
        season_encoded = pd.get_dummies(data_task_4['season'], drop_first=True)
        data_encoded = pd.concat([data_task_4, season_encoded], axis=1)

        for season in ['Spring', 'Summer', 'Winter']:
            if season not in data_encoded.columns:
                data_encoded[season] = 0

        X = data_encoded[['Spring', 'Summer', 'Winter']].astype(int)
        y = data_encoded['score']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

    else:
        st.warning("Please load the dataset first.")

elif section == "► 4. Post-Release Market Impact Analysis":
    st.title("4. Post-Release Market Impact Analysis")
    st.subheader(
        "4.1. Classification of games into \"Editor's Choice\" and \"Not Editor's Choice\" based on their rating and genre")

    if 'data' in st.session_state:
        subset_task_2 = st.session_state.data[['score', 'genre', 'editors_choice']]
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_genre = encoder.fit_transform(subset_task_2[['genre']])
        encoded_genre_df = pd.DataFrame(encoded_genre, columns=encoder.get_feature_names_out(['genre']))
        data_encoded = pd.concat([subset_task_2.reset_index(drop=True), encoded_genre_df], axis=1).drop('genre', axis=1)
        data_encoded['editors_choice'] = data_encoded['editors_choice'].map({'Y': 1, 'N': 0})
        X = data_encoded.drop('editors_choice', axis=1)
        y = data_encoded['editors_choice']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        st.markdown("### Logistic Regression Model Training and Hyperparameter Tuning")
        max_iter = st.slider("Max Iterations for Logistic Regression", 100, 10000, 1000, step=100)
        C = st.slider("Regularization strength (C)", 0.01, 1.0, 0.5, step=0.01)

        logreg = LogisticRegression(max_iter=max_iter, C=C)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy of the Logistic Regression Model: {accuracy * 100:.2f}%")

        st.markdown("### Interactive Prediction")
        input_score = st.number_input("Enter a game score", min_value=0.0, max_value=10.0, value=5.0)
        selected_genres = st.multiselect("Select game genres", options=encoder.get_feature_names_out(['genre']))
        if st.button("Predict Editor's Choice"):
            input_data = pd.DataFrame([input_score], columns=['score'])
            for genre in encoder.get_feature_names_out(['genre']):
                input_data[genre] = 1 if genre in selected_genres else 0
            prediction = logreg.predict(input_data)
            result = "Editor's Choice" if prediction[0] == 1 else "Not Editor's Choice"
            st.write(f"The game is predicted to be: {result}")

    else:
        st.warning("Please load the dataset first.")

elif section == "► 5. Enhancing User Engagement and Sales Strategies":
    st.title("5. Enhancing User Engagement and Sales Strategies")
    if 'data' in st.session_state:
        data = st.session_state.data

        data_task_3 = data.copy()

        encoder = OneHotEncoder()
        encoded_features = encoder.fit_transform(data_task_3[['genre', 'platform']])
        encoded_df = pd.DataFrame(encoded_features.toarray(),
                                  columns=encoder.get_feature_names_out(['genre', 'platform']))

        scaler = MinMaxScaler()
        data_task_3['normalized_score'] = scaler.fit_transform(data_task_3[['score']])

        final_data = pd.concat([data_task_3, encoded_df], axis=1)
        features = final_data.iloc[:, -encoded_df.shape[1]:]
        features['normalized_score'] = final_data['normalized_score']
        platform_columns = [col for col in features.columns if "platform_" in col]
        features[platform_columns] = features[platform_columns] * 2

        def recommend_games(game_title, platform, num_recommendations=5):
            if game_title not in final_data['title'].values:
                return []

            subset_data = final_data[final_data['platform'] == platform]
            if subset_data.shape[0] == 0:
                return []

            subset_features = subset_data[features.columns]
            game_features = final_data[final_data['title'] == game_title][features.columns].values
            similarity_scores = cosine_similarity(game_features, subset_features)[0]
            sorted_game_indices = np.argsort(similarity_scores)[::-1]
            recommended_games = subset_data.iloc[sorted_game_indices[:num_recommendations + 1]]['title'].tolist()
            recommended_games = [game for game in recommended_games if game != game_title]

            return recommended_games[:num_recommendations]

        st.markdown("### Get Game Recommendations")
        game_to_recommend = st.selectbox("Select a game:", final_data['title'].unique())

        available_platforms = final_data[final_data['title'] == game_to_recommend]['platform'].unique()
        platform_for_recommendation = st.selectbox("Select a platform:", available_platforms)

        recommendations = recommend_games(game_to_recommend, platform_for_recommendation)

        st.markdown(f"Recommendations for **{game_to_recommend}** on platform **{platform_for_recommendation}**:")

        if len(recommendations) == 0:
            st.warning(f"No recommended games for {game_to_recommend}.")
        else:
            for idx, title in enumerate(recommendations, 1):
                st.write(f"{idx}. {title}")
    else:
        st.warning("Please load the dataset first.")
