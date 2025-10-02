#!/usr/bin/env node

/**
 * Privium Website Monitoring Script
 * Checks that all pages are accessible and match expected content
 * Run with: node monitor-site.js
 * Or schedule with cron (every 10 minutes)
 */

const https = require('https');
const fs = require('fs');

const SITE_URL = 'https://privium.com.au';
const PAGES = [
    { path: '/', name: 'Home', checkText: 'Beautiful Lawns for Every Lifestyle' },
    { path: '/services.html', name: 'Services', checkText: 'Garden Care Services Perth' },
    { path: '/about.html', name: 'About', checkText: 'About Privium Garden Services' },
    { path: '/quote.html', name: 'Contact', checkText: 'Get Quote' },
    { path: '/client-portal/login.html', name: 'Client Portal Login', checkText: 'Client Portal' }
];

// Color codes for terminal output
const colors = {
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    reset: '\x1b[0m'
};

function fetchPage(url) {
    return new Promise((resolve, reject) => {
        https.get(url, {
            headers: {
                'User-Agent': 'Privium-Monitor/1.0'
            }
        }, (res) => {
            let data = '';

            res.on('data', (chunk) => {
                data += chunk;
            });

            res.on('end', () => {
                resolve({
                    statusCode: res.statusCode,
                    body: data
                });
            });
        }).on('error', (err) => {
            reject(err);
        });
    });
}

async function checkPage(page) {
    const url = SITE_URL + page.path;

    try {
        const result = await fetchPage(url);

        if (result.statusCode !== 200) {
            return {
                success: false,
                page: page.name,
                url: url,
                error: `HTTP ${result.statusCode}`,
                timestamp: new Date().toISOString()
            };
        }

        if (!result.body.includes(page.checkText)) {
            return {
                success: false,
                page: page.name,
                url: url,
                error: `Expected text "${page.checkText}" not found`,
                timestamp: new Date().toISOString()
            };
        }

        return {
            success: true,
            page: page.name,
            url: url,
            timestamp: new Date().toISOString()
        };
    } catch (error) {
        return {
            success: false,
            page: page.name,
            url: url,
            error: error.message,
            timestamp: new Date().toISOString()
        };
    }
}

async function monitorSite() {
    console.log(`\n${colors.yellow}=== Privium Website Monitor ===${colors.reset}`);
    console.log(`${new Date().toLocaleString()}\n`);

    const results = [];

    for (const page of PAGES) {
        const result = await checkPage(page);
        results.push(result);

        if (result.success) {
            console.log(`${colors.green}✓${colors.reset} ${result.page.padEnd(25)} ${result.url}`);
        } else {
            console.log(`${colors.red}✗${colors.reset} ${result.page.padEnd(25)} ${result.url}`);
            console.log(`  ${colors.red}Error: ${result.error}${colors.reset}`);
        }
    }

    // Log to file
    const logEntry = {
        timestamp: new Date().toISOString(),
        results: results
    };

    const logFile = 'monitor-log.json';
    let logs = [];

    if (fs.existsSync(logFile)) {
        try {
            logs = JSON.parse(fs.readFileSync(logFile, 'utf8'));
        } catch (e) {
            logs = [];
        }
    }

    logs.push(logEntry);

    // Keep only last 1000 entries (roughly 1 week at 10min intervals)
    if (logs.length > 1000) {
        logs = logs.slice(-1000);
    }

    fs.writeFileSync(logFile, JSON.stringify(logs, null, 2));

    // Summary
    const failures = results.filter(r => !r.success).length;
    console.log(`\n${failures === 0 ? colors.green : colors.red}Summary: ${results.length - failures}/${results.length} pages OK${colors.reset}\n`);

    // Exit with error code if any failures
    if (failures > 0) {
        process.exit(1);
    }
}

// Run the monitor
monitorSite().catch(err => {
    console.error(`${colors.red}Fatal error: ${err.message}${colors.reset}`);
    process.exit(1);
});
