const crypto = require('crypto');

// Same secret key as used in login.html
const SECRET_KEY = 'PRIVIUM_GARDEN_SERVICES_2025_SECRET_KEY_v1';

// The HTML content to encrypt (from guide-scalping.html lines 469-674)
const contentHTML = `
    <div class="container">
        <header>
            <img src="privium-long-logo.png" alt="Privium Garden Services" class="logo">
            <div class="header-text">
                <h1>Scalping Service</h1>
                <p class="subtitle">Aftercare Guide & Recovery Timeline</p>
            </div>
        </header>

        <div class="content">
            <div class="alert-box">
                <strong>⚠️ Important:</strong> Your lawn will look brown and stressed immediately after scalping. This is completely normal and necessary for healthy regrowth. Full recovery typically takes 2-3 weeks with proper care.
            </div>

            <h2>What is Scalping?</h2>
            <p>Scalping is an aggressive mowing technique that removes thatch buildup and stimulates vigorous new growth. We've cut your lawn to 10-15mm height, removing up to 70% of the leaf blade to expose stems and promote fresh, healthy regrowth.</p>

            <h2>Recovery Timeline</h2>
            <div class="timeline">
                <div class="timeline-item">
                    <div class="timeline-badge">Day 1-3</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Initial Shock Phase</div>
                        <ul>
                            <li>Lawn appears brown/yellow</li>
                            <li>Stems and thatch visible</li>
                            <li>Begin consistent watering schedule</li>
                            <li>No foot traffic</li>
                        </ul>
                    </div>
                </div>

                <div class="timeline-item">
                    <div class="timeline-badge">Day 4-7</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Early Recovery</div>
                        <ul>
                            <li>First signs of green shoots</li>
                            <li>Continue regular watering</li>
                            <li>Apply liquid fertiliser (optional)</li>
                            <li>Minimal foot traffic only</li>
                        </ul>
                    </div>
                </div>

                <div class="timeline-item">
                    <div class="timeline-badge">Week 2</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Active Growth</div>
                        <ul>
                            <li>Significant green-up visible</li>
                            <li>New leaf growth throughout</li>
                            <li>First light mow when reaches 40mm</li>
                            <li>Light foot traffic permitted</li>
                        </ul>
                    </div>
                </div>

                <div class="timeline-item">
                    <div class="timeline-badge">Week 3</div>
                    <div class="timeline-content">
                        <div class="timeline-title">Full Recovery</div>
                        <ul>
                            <li>Dense, healthy appearance</li>
                            <li>Resume normal mowing schedule</li>
                            <li>Return to regular watering</li>
                            <li>Normal use can resume</li>
                        </ul>
                    </div>
                </div>
            </div>

            <h2>Critical Watering Schedule</h2>
            <table class="schedule-table">
                <thead>
                    <tr>
                        <th>Period</th>
                        <th>Frequency</th>
                        <th>Duration</th>
                        <th>Time of Day</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Day 1-3</strong></td>
                        <td>2x daily</td>
                        <td>10-15 min</td>
                        <td>7am, 3pm</td>
                        <td>Keep soil moist, not waterlogged</td>
                    </tr>
                    <tr>
                        <td><strong>Day 4-7</strong></td>
                        <td>Daily</td>
                        <td>15-20 min</td>
                        <td>Early morning</td>
                        <td>Deep watering to encourage roots</td>
                    </tr>
                    <tr>
                        <td><strong>Week 2</strong></td>
                        <td>Every 2 days</td>
                        <td>20-25 min</td>
                        <td>Early morning</td>
                        <td>Adjust for rainfall</td>
                    </tr>
                    <tr>
                        <td><strong>Week 3+</strong></td>
                        <td>2-3x weekly</td>
                        <td>25-30 min</td>
                        <td>Early morning</td>
                        <td>Return to normal schedule</td>
                    </tr>
                </tbody>
            </table>

            <h2>Do's and Don'ts</h2>
            <div class="dos-donts">
                <div class="dos">
                    <h3>✓ DO:</h3>
                    <ul>
                        <li>Water consistently as scheduled</li>
                        <li>Apply liquid fertiliser after Day 5</li>
                        <li>Monitor for dry patches</li>
                        <li>Keep pets off for first week</li>
                        <li>Mow high (40mm) for first cut</li>
                        <li>Gradually lower mowing height</li>
                    </ul>
                </div>
                <div class="donts">
                    <h3>✗ DON'T:</h3>
                    <ul>
                        <li>Skip watering days</li>
                        <li>Mow too early (wait for 40mm)</li>
                        <li>Apply heavy foot traffic</li>
                        <li>Over-fertilize</li>
                        <li>Water in evening</li>
                        <li>Panic about brown appearance</li>
                    </ul>
                </div>
            </div>

            <div class="visual-guide">
                <h3>Visual Recovery Progress</h3>
                <div class="photo-grid">
                    <div class="photo-item">
                        <div class="photo-placeholder">Day 1: Brown/Yellow</div>
                        <div class="photo-label">Immediately After</div>
                    </div>
                    <div class="photo-item">
                        <div class="photo-placeholder">Week 1: Green Shoots</div>
                        <div class="photo-label">Early Recovery</div>
                    </div>
                    <div class="photo-item">
                        <div class="photo-placeholder">Week 3: Full Green</div>
                        <div class="photo-label">Fully Recovered</div>
                    </div>
                </div>
            </div>

            <div style="page-break-inside: avoid; -webkit-column-break-inside: avoid; break-inside: avoid; display: block;">
            <h2>Fertilizer Recommendations</h2>
            <div class="two-column">
                <div class="info-card">
                    <h3>Week 1</h3>
                    <p><strong>Day 5-7:</strong> Apply liquid fertiliser or seaweed solution to stimulate growth. Use half-strength mixture to avoid burning.</p>
                </div>
                <div class="info-card">
                    <h3>Week 2-3</h3>
                    <p><strong>Day 10-14:</strong> Apply balanced NPK fertiliser at recommended rates. Water in well after application.</p>
                </div>
            </div>

            <div style="page-break-inside: avoid; -webkit-column-break-inside: avoid; break-inside: avoid; display: block;">
                <h2>Troubleshooting Common Issues</h2>
                <div class="info-card">
                    <p><strong>Uneven regrowth:</strong> Check sprinkler coverage and hand-water dry areas.</p>
                    <p><strong>Slow recovery:</strong> Apply liquid fertiliser and ensure adequate watering.</p>
                    <p><strong>Yellow patches:</strong> May indicate nutrient deficiency - apply iron supplement.</p>
                    <p><strong>Weeds appearing:</strong> Normal after scalping - hand remove or spot spray after Week 2.</p>
                </div>
            </div>

            <div class="success-box">
                <strong>Success Tip:</strong> The key to successful recovery is consistent watering. Set reminders on your phone for the first week to ensure you don't miss watering times. Your lawn's recovery depends on it!
            </div>

            <div class="contact-bar">
                <div class="contact-item">
                    <span>📞</span>
                    <strong>0490 841 667</strong>
                </div>
                <div class="contact-item">
                    <span>👤</span>
                    <span>Ask for Jarod</span>
                </div>
                <div class="contact-item">
                    <span>✉️</span>
                    <span>office@privium.com.au</span>
                </div>
            </div>
        </div>

        <footer>
            <p>PRIVIUM GARDEN SERVICES PTY LTD • Professional Lawn Care Perth Metro • © 2025</p>
        </footer>
    </div>
`;

// Create a SHA-256 hash of the secret key to use as encryption key
const key = crypto.createHash('sha256').update(SECRET_KEY).digest();

// Generate a random IV (Initialization Vector)
const iv = crypto.randomBytes(16);

// Create cipher
const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);

// Encrypt the content
let encrypted = cipher.update(contentHTML, 'utf8', 'base64');
encrypted += cipher.final('base64');

// Combine IV and encrypted data (IV needs to be stored with the encrypted data for decryption)
const encryptedWithIV = iv.toString('base64') + ':' + encrypted;

console.log('Encrypted content (ready to paste into guide-scalping.html):');
console.log('');
console.log(encryptedWithIV);
console.log('');
console.log('Length:', encryptedWithIV.length, 'characters');