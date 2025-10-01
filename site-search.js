// Privium Garden Services - Site Search
// Search data index for all main pages

const searchData = [
    // Homepage
    {
        page: "Home",
        url: "index.html",
        title: "Privium Garden Services | Premium Garden Care Perth",
        sections: [
            {
                heading: "Perth's Trusted Garden Care Professionals",
                anchor: "",
                content: "Premium lawn maintenance hedge trimming garden services Cottesloe Dalkeith South Perth surrounding suburbs professional garden care quality reliability outstanding results"
            },
            {
                heading: "Subscription Garden Care",
                anchor: "",
                content: "regular scheduled garden maintenance lawn mowing hedges seasonal care complete garden ecosystem management Perth premium suburbs"
            },
            {
                heading: "Lawn Renovation Services",
                anchor: "",
                content: "scalping aeration topdressing complete lawn transformation healthy lawn Perth sandy soils"
            },
            {
                heading: "Garden Services",
                anchor: "",
                content: "hedge trimming mulching pruning weed treatment fertilising pest disease control irrigation garden bed renovation"
            },
            {
                heading: "Specialised Services",
                anchor: "",
                content: "garden health assessment expert solutions pest issues irrigation concerns plant health knowledge tools"
            }
        ]
    },
    // Services Page
    {
        page: "Services",
        url: "services.html",
        title: "Garden Care Services Perth | Lawn Mowing, Hedge Trimming & More",
        sections: [
            {
                heading: "Subscription Garden Care",
                anchor: "#subscription",
                content: "regular scheduled maintenance fortnightly monthly visits lawn mowing hedging seasonal fertilising complete garden care professional service Perth"
            },
            {
                heading: "Lawn Renovation",
                anchor: "#lawn-care",
                content: "scalping only scalping aeration scalping topdressing complete renovation transform lawn remove thatch stimulate growth Perth"
            },
            {
                heading: "Garden Maintenance",
                anchor: "#garden-maintenance",
                content: "hedge trimming shaping mulch application pruning shrubs trees weed control treatment fertiliser fertilising lawn garden beds pest disease management irrigation reticulation repairs"
            },
            {
                heading: "Specialised Surfaces",
                anchor: "#specialised",
                content: "synthetic grass maintenance tennis courts natural grass synthetic surfaces garden beds renovation soft landscaping"
            },
            {
                heading: "Expert Services",
                anchor: "#packages",
                content: "new lawn installation soil testing preparation establishment varieties Perth couch buffalo kikuyu"
            }
        ]
    },
    // About Page
    {
        page: "About",
        url: "about.html",
        title: "About Privium Garden Services | Perth's Premium Garden Care Professionals",
        sections: [
            {
                heading: "Established 2009",
                anchor: "",
                content: "quality service experienced team servicing premium suburbs Perth garden care professionals trusted reliable"
            },
            {
                heading: "Comprehensive Subscription Service",
                anchor: "",
                content: "integrated lawn care tree shrub maintenance pest management irrigation optimisation seasonal adjustments ecosystem complete care"
            },
            {
                heading: "Professional Team Equipment",
                anchor: "",
                content: "specialised equipment tripod ladders large hedges cylinder mowers golf-cut lawns professional-grade application bio-stimulants treatments"
            },
            {
                heading: "Detailed Documentation",
                anchor: "",
                content: "customised approach seasonal assessment reports document work track garden changes monitor nutrient pH levels optimise watering cycles recommendations improvement"
            },
            {
                heading: "Convenience That Works",
                anchor: "",
                content: "streamlined process initial consultation ongoing service reminders waste removal minor irrigation repairs fruit collection bird bath filling details"
            },
            {
                heading: "Service Areas",
                anchor: "",
                content: "Cottesloe Dalkeith Nedlands Claremont Peppermint Grove Mosman Park Swanbourne City Beach Floreat Shenton Park Mount Claremont South Perth Como Kensington Applecross Mount Pleasant Salter Point Attadale Bicton"
            }
        ]
    },
    // Quote/Contact Page
    {
        page: "Contact",
        url: "quote.html",
        title: "Get Quote | Free Garden Care Assessment Perth",
        sections: [
            {
                heading: "Request Free Quote",
                anchor: "",
                content: "free quote garden care services Perth professional lawn mowing hedge trimming maintenance quick response competitive rates assessment"
            },
            {
                heading: "Contact Details",
                anchor: "",
                content: "Call Jarod 0490 841 667 office@privium.com.au Perth garden services professional care"
            }
        ]
    }
];

// Search function
function performSearch(query) {
    if (!query || query.trim().length < 2) {
        return [];
    }

    query = query.toLowerCase().trim();
    const results = [];

    searchData.forEach(page => {
        let pageScore = 0;
        let matchedSections = [];

        // Check page title
        if (page.title.toLowerCase().includes(query)) {
            pageScore += 10;
        }

        // Check each section
        page.sections.forEach(section => {
            const headingMatch = section.heading.toLowerCase().includes(query);
            const contentMatch = section.content.toLowerCase().includes(query);

            if (headingMatch || contentMatch) {
                const sectionScore = headingMatch ? 5 : 1;
                pageScore += sectionScore;

                // Create snippet
                const words = section.content.toLowerCase().split(' ');
                const queryIndex = words.findIndex(word => word.includes(query));
                let snippet = section.heading;

                if (contentMatch && queryIndex >= 0) {
                    const start = Math.max(0, queryIndex - 5);
                    const end = Math.min(words.length, queryIndex + 10);
                    snippet = '...' + words.slice(start, end).join(' ') + '...';
                }

                matchedSections.push({
                    heading: section.heading,
                    snippet: snippet,
                    score: sectionScore,
                    anchor: section.anchor || ''
                });
            }
        });

        if (pageScore > 0) {
            results.push({
                page: page.page,
                url: page.url,
                title: page.title,
                score: pageScore,
                sections: matchedSections.sort((a, b) => b.score - a.score).slice(0, 2)
            });
        }
    });

    // Sort by relevance
    return results.sort((a, b) => b.score - a.score);
}

// Highlight search terms in text
function highlightText(text, query) {
    if (!query) return text;

    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<mark style="background: #ffeb3b; padding: 2px 4px; border-radius: 2px;">$1</mark>');
}

// Toggle search visibility
function toggleSearch() {
    const searchInput = document.getElementById('siteSearch');
    const searchToggle = document.getElementById('searchToggle');
    const searchResults = document.getElementById('searchResults');

    if (searchInput.style.display === 'none') {
        searchInput.style.display = 'block';
        searchToggle.style.display = 'none';
        searchInput.focus();
    } else {
        searchInput.style.display = 'none';
        searchToggle.style.display = 'flex';
        searchResults.style.display = 'none';
        searchInput.value = '';
    }
}

// Initialize search UI
function initializeSearch() {
    const searchInput = document.getElementById('siteSearch');
    const searchResults = document.getElementById('searchResults');

    if (!searchInput || !searchResults) return;

    let searchTimeout;

    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);

        searchTimeout = setTimeout(() => {
            const query = e.target.value;

            if (query.length < 2) {
                searchResults.style.display = 'none';
                return;
            }

            const results = performSearch(query);
            displaySearchResults(results, query);
        }, 300);
    });

    // Click outside to close
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            const searchToggle = document.getElementById('searchToggle');
            searchResults.style.display = 'none';
            searchInput.style.display = 'none';
            searchToggle.style.display = 'flex';
            searchInput.value = '';
        }
    });

    // Show results on focus if there's a query
    searchInput.addEventListener('focus', () => {
        if (searchInput.value.length >= 2) {
            const results = performSearch(searchInput.value);
            displaySearchResults(results, searchInput.value);
        }
    });

    // Close on Escape key
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            toggleSearch();
        }
    });
}

// Display search results
function displaySearchResults(results, query) {
    const searchResults = document.getElementById('searchResults');

    if (results.length === 0) {
        searchResults.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #666;">
                No results found for "${query}"
            </div>
        `;
        searchResults.style.display = 'block';
        return;
    }

    let html = '';

    results.forEach(result => {
        // Use the first matched section's anchor if available
        const topSection = result.sections[0];
        const targetUrl = topSection && topSection.anchor ? result.url + topSection.anchor : result.url;

        html += `
            <a href="${targetUrl}" class="search-result-item" style="display: block; padding: 15px; border-bottom: 1px solid #eee; text-decoration: none; transition: background 0.2s;" onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='white'">
                <div style="font-weight: 600; color: #244419; margin-bottom: 5px; font-size: 14px;">
                    ${result.page}
                </div>
                ${result.sections.slice(0, 1).map(section => `
                    <div style="font-size: 12px; color: #666; line-height: 1.4;">
                        ${highlightText(section.snippet, query)}
                    </div>
                `).join('')}
            </a>
        `;
    });

    searchResults.innerHTML = html;
    searchResults.style.display = 'block';
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSearch);
} else {
    initializeSearch();
}
