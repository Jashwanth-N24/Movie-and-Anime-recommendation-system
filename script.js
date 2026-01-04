document.addEventListener('DOMContentLoaded', () => {
    const movieInput = document.getElementById('movieInput');
    const movieList = document.getElementById('movieList');
    const searchBtn = document.getElementById('searchBtn');
    const cardsGrid = document.getElementById('cardsGrid');
    const resultsSection = document.getElementById('resultsSection');
    const introState = document.getElementById('introState');
    const loading = document.getElementById('loading');
    const sourceTitle = document.getElementById('sourceTitle');

    // 1. Fetch available movies on load
    fetch('/api/movies')
        .then(res => res.json())
        .then(data => {
            if (data.movies) {
                // Limit the datalist size for performance/usability if needed, 
                // but standard datalists handle thousands fairly well.
                data.movies.forEach(movie => {
                    const option = document.createElement('option');
                    option.value = movie;
                    movieList.appendChild(option);
                });
            }
        })
        .catch(err => console.error("Error loading movies:", err));

    // 2. Search Functionality
    function performSearch() {
        const title = movieInput.value;
        if (!title) return;

        // UI Updates
        introState.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loading.classList.remove('hidden');
        cardsGrid.innerHTML = ''; // Clear previous

        fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title: title })
        })
            .then(res => res.json())
            .then(data => {
                loading.classList.add('hidden');

                if (data.error) {
                    alert(data.error);
                    introState.classList.remove('hidden');
                    return;
                }

                sourceTitle.textContent = data.source_title;
                renderCards(data.recommendations);
                resultsSection.classList.remove('hidden');
            })
            .catch(err => {
                console.error(err);
                loading.classList.add('hidden');
                alert("Something went wrong. Please try again.");
                introState.classList.remove('hidden');
            });
    }

    // Event Listeners
    searchBtn.addEventListener('click', performSearch);

    movieInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });

    // Helper: Generate a consistent HSL color from string
    function stringToColor(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        const h = Math.abs(hash % 360);
        return `hsl(${h}, 70%, 65%)`; // Soft pastel colors
    }

    // Helper: Generate gradient based on title
    function generateGradient(title) {
        const c1 = stringToColor(title);
        const c2 = stringToColor(title.split('').reverse().join(''));
        return `linear-gradient(135deg, ${c1}, ${c2})`;
    }

    // Helper: Get Initials
    function getInitials(title) {
        return title.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
    }

    // 3. Render Cards
    function renderCards(movies) {
        movies.forEach((movie, index) => {
            const card = document.createElement('div');
            card.className = 'card';
            card.style.animation = `fadeIn 0.5s ease backwards ${index * 0.1}s`;

            // Prepare Poster Data
            const gradient = generateGradient(movie.title);
            const initials = getInitials(movie.title);

            // Genre Badges (Max 3)
            const genresHtml = movie.genres.slice(0, 3).map(g => `<span class="badge genre">${g}</span>`).join('');

            // Platform Badges
            let platformsHtml = '';
            if (movie.platforms && movie.platforms.length > 0) {
                platformsHtml = movie.platforms.map(p => {
                    let className = 'other-platform';
                    if (p.includes('Netflix')) className = 'netflix';
                    if (p.includes('Prime')) className = 'prime';
                    if (p.includes('Disney')) className = 'disney';
                    if (p.includes('Hulu')) className = 'hulu';
                    return `<span class="platform-badge ${className}">${p}</span>`;
                }).join('');
            }

            // Cast
            const castHtml = movie.cast && movie.cast.length > 0
                ? `<div class="cast-info"><small><strong>Cast:</strong> ${movie.cast.join(', ')}</small></div>`
                : '';

            card.innerHTML = `
                <div class="card-body">
                    <h3 class="card-title" style="margin-bottom:0.5rem; font-size:1.1rem;">${movie.display_title}</h3>
                    <div class="card-meta">
                        ${genresHtml}
                    </div>
                    <p class="overview">${movie.overview}</p>
                    ${castHtml}
                    <div class="platforms">
                        ${platformsHtml}
                    </div>
                </div>
            `;

            cardsGrid.appendChild(card);
        });
    }
});

