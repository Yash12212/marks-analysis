from flask import Flask, render_template_string, request, jsonify, send_file, g
import pandas as pd
import sqlite3
import numpy as np
import io
from datetime import datetime

app = Flask(__name__)

# Database configuration
DATABASE = 'scores.db'
HISTORY_LIMIT = 10  # Limit for the number of history entries to retrieve

def get_db():
    """
    Get database connection, creating it if it doesn't exist in the application context.
    Uses sqlite3.Row for dictionary-like row access.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row  # Return rows as dictionaries
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """
    Close the database connection when the application context is torn down.
    """
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """
    Initialize the database by creating the 'scores' table and index if they don't exist.
    """
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS scores
                     (id INTEGER PRIMARY KEY,
                      data TEXT,
                      mean REAL,
                      std_dev REAL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON scores (timestamp)') # Index for efficient history retrieval
        db.commit()

init_db()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analyzer Pro</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" integrity="sha512-9usAa10IRO0HhonpyAIVpjrylPvoDwiPUiKdWk5t3PyolY1cOd4DSE0Ga+ri4AuTroPR5aQvXU9xC6qOPnzFeg==" crossorigin="anonymous" referrerpolicy="no-referrer" />

    <style>
        .history-details {
            display: none;
            padding-top: 0.5rem;
        }
        .history-details.active {
            display: block;
        }
        .chart-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        /* Chart Styling - Ensure chart takes available width and has a minimum height */
        #chartContainer {
            min-height: 300px; /* Minimum height for the chart container */
        }
        #chart {
            width: 100%; /* Make chart canvas take full width of its container */
            height: auto; /* Height adjusts based on aspect ratio and container */
            display: block; /* Prevents extra space below canvas */
        }
    </style>
</head>
<body class="bg-gray-100 font-sans">
    <div class="container mx-auto p-4">
        <h1 class="text-3xl font-bold text-blue-600 mb-6 text-center">📈 Data Analyzer Pro</h1>

        <section class="upload-section bg-white shadow-md rounded-lg p-6 mb-6">
            <h3 class="text-xl font-semibold text-gray-700 mb-4"><i class="fas fa-upload mr-2"></i> Upload CSV File</h3>
            <input type="file" id="csvFile" accept=".csv" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-700">
            <p class="mt-2 text-sm text-gray-500">Supports CSV files with numerical data in any column</p>
            <div class="error-message text-red-500 mt-2 hidden" id="fileError"></div>
        </section>

        <section class="upload-section bg-white shadow-md rounded-lg p-6 mb-6">
            <h3 class="text-xl font-semibold text-gray-700 mb-4"><i class="fas fa-keyboard mr-2"></i> Or Enter Data Manually</h3>
            <textarea id="manualData" placeholder="Example: 85, 90, 78.5, 92, 88.2" class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md"></textarea>
            <p class="mt-2 text-sm text-gray-500">Separate numbers with commas. Decimals allowed.</p>
            <button onclick="analyzeManualData()" class="analyze-btn bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-4"><i class="fas fa-chart-bar mr-2"></i> Analyze Manual Input</button>
            <div class="error-message text-red-500 mt-2 hidden" id="manualError"></div>
        </section>

        <section class="stats bg-white shadow-md rounded-lg p-6 mb-6" id="results">
            <!-- Results will be displayed here -->
            <p class="text-gray-600">No analysis yet. Upload a CSV or enter data manually.</p>
        </section>

        <section id="chartContainer" class="bg-white shadow-md rounded-lg p-6 mb-6">
            <canvas id="chart" role="img" aria-label="Data Analysis Chart"></canvas>
            <div class="chart-controls mt-4 flex justify-end space-x-2">
                <button class="chart-btn reset-zoom-btn bg-gray-200 hover:bg-gray-300 text-gray-700 font-semibold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed" onclick="resetZoom()" disabled><i class="fas fa-sync-alt mr-2"></i> Reset Zoom</button>
                <button class="chart-btn download-chart-btn bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed" onclick="downloadChart()" disabled><i class="fas fa-download mr-2"></i> Download Chart</button>
            </div>
        </section>


        <section class="history bg-white shadow-md rounded-lg p-6 mb-6">
            <h2 class="text-xl font-semibold text-gray-700 mb-4"><i class="fas fa-history mr-2"></i> Analysis History <span id="historyCount" class="text-sm text-gray-500">(0 entries)</span></h2>
            <div id="historyList">
                <p class="text-gray-500">History will be displayed here once analysis is performed.</p>
            </div>
            <div class="mt-4">
                <button class="delete-all-history-btn bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded" onclick="confirmDeleteAll()" disabled id="deleteAllHistoryBtn" title="Delete all history (disabled when empty)"><i class="fas fa-trash-alt mr-2"></i> Delete All History</button>
                <a href="/download_db" class="download-db-btn bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded ml-2" title="Download the analysis history database"><i class="fas fa-database mr-2"></i> Download Database</a>
            </div>
        </section>

        <div class="loading hidden justify-center items-center space-x-2 mb-4" id="loading">
            <div class="spinner border-4 border-blue-500 border-t-transparent rounded-full w-8 h-8 animate-spin"></div>
            <p id="loadingText" class="text-blue-600">Analyzing Data...</p>
        </div>
    </div>

    <footer class="bg-gray-200 text-center p-4 mt-8">
        <p class="text-gray-600 text-sm">Data Analyzer Pro © 2023</p>
    </footer>

    <script>
        let currentChart = null;
        let chartData = null;
        const loading = document.getElementById('loading');
        const loadingText = document.getElementById('loadingText');
        const resultsDiv = document.getElementById('results');
        const chartContainer = document.getElementById('chartContainer');
        const chartCanvas = document.getElementById('chart');
        const historyListDiv = document.getElementById('historyList');
        const historyCountSpan = document.getElementById('historyCount');
        const deleteAllHistoryBtn = document.getElementById('deleteAllHistoryBtn');
        const resetZoomBtn = document.querySelector('.reset-zoom-btn');
        const downloadChartBtn = document.querySelector('.download-chart-btn');


        function showLoading(message) {
            loadingText.textContent = message;
            loading.classList.remove('hidden');
            loading.classList.add('flex');
        }

        function hideLoading() {
            loading.classList.add('hidden');
            loading.classList.remove('flex');
        }

        async function analyzeManualData() {
            const manualInput = document.getElementById('manualData').value;
            const errorDiv = document.getElementById('manualError');
            errorDiv.classList.add('hidden');

            // Clean and validate input
            const cleanedInput = manualInput.replace(/[^0-9.,]/g, '');
            const data = cleanedInput.split(/,\s*/)
                .map(x => parseFloat(x))
                .filter(x => !isNaN(x));

            if (data.length < 2) {
                errorDiv.textContent = 'Please enter at least 2 valid numbers';
                errorDiv.classList.remove('hidden');
                return;
            }

            showLoading('Analyzing Manual Input...');

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data })
                });

                if (!response.ok) {
                    const error = await response.json(); // Try to get JSON error response
                    throw new Error(error.error || `HTTP error! status: ${response.status}`); // Fallback to generic error if no JSON error
                }

                const result = await response.json();
                updateUI(result);
                await loadHistory();
            } catch (error) {
                errorDiv.textContent = `Error: ${error.message}`;
                errorDiv.classList.remove('hidden');
            } finally {
                hideLoading();
            }
        }

        function updateUI(result) {
            resultsDiv.innerHTML = `
                <h3 class="text-xl font-semibold text-gray-700 mb-2">Analysis #${result.id}</h3>
                <p class="text-gray-600"><i class="fas fa-calculator mr-1"></i> Mean: <span class="font-medium">${result.stats.mean.toFixed(2)}</span></p>
                <p class="text-gray-600"><i class="fas fa-chart-line mr-1"></i> Standard Deviation: <span class="font-medium">${result.stats.std_dev.toFixed(2)}</span></p>
                <p class="text-gray-600"><i class="fas fa-chart-area mr-1"></i> Trend Equation: <span class="font-medium">y = ${result.stats.trend[0].toFixed(2)}x + ${result.stats.trend[1].toFixed(2)}</span></p>
            `;
            chartContainer.classList.remove('hidden'); // Ensure chart container is visible
            updateChart(result.data, result.stats);
        }

        function updateChart(data, stats) {
            const ctx = chartCanvas.getContext('2d');
            if (currentChart) currentChart.destroy();

            chartData = data; // Store data for download

            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map((_, i) => i+1),
                    datasets: [{
                        label: 'Data Values',
                        data: data,
                        borderColor: '#3b82f6',
                        tension: 0.1,
                        pointRadius: 3
                    }, {
                        label: 'Trend Line',
                        data: stats.trend_values,
                        borderColor: '#ef4444',
                        borderDash: [5, 5],
                        tension: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        zoom: {
                            zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' },
                            pan: { enabled: true, mode: 'xy' }
                        },
                        title: {
                            display: true,
                            text: 'Data Points with Trend Line',
                            padding: 10,
                            font: {
                                size: 16
                            }
                        },
                        legend: {
                            position: 'bottom',
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Data Point Index' } },
                        y: { title: { display: true, text: 'Value' } }
                    }
                }
            });

            // --- DEBUGGING LOGS - TIME SERIES ---
            const logInterval = setInterval(() => {
                console.log("--- Chart Dimensions (Time Series Log) ---");
                console.log("Chart Canvas Dimensions:");
                console.log("offsetWidth:", chartCanvas.offsetWidth, "offsetHeight:", chartCanvas.offsetHeight);
                console.log("clientWidth:", chartCanvas.clientWidth, "clientHeight:", chartCanvas.clientHeight);
                console.log("Container Dimensions:");
                console.log("offsetWidth:", chartContainer.offsetWidth, "offsetHeight:", chartContainer.offsetHeight);
                console.log("clientWidth:", chartContainer.clientWidth, "clientHeight:", chartContainer.clientHeight);
                console.log("Chart Options:", currentChart.config.options);
            }, 1000); // Log every 1 second (adjust interval as needed)

            // Stop logging when chart is destroyed (e.g., on new analysis)
            if (currentChart) {
                const originalDestroy = currentChart.destroy;
                currentChart.destroy = function() {
                    clearInterval(logInterval); // Clear the interval when destroying chart
                    originalDestroy.apply(this, arguments);
                };
            }
            // --- END DEBUGGING LOGS ---


            resetZoomBtn.disabled = false;
            downloadChartBtn.disabled = false;
        }

        function resetZoom() {
            if (currentChart) {
                currentChart.resetZoom();
            }
        }

        function downloadChart() {
            if (currentChart) {
                const link = document.createElement('a');
                link.download = 'data_analysis_chart.png';
                link.href = chartCanvas.toDataURL();
                link.click();
            }
        }

        async function loadHistory() {
            try {
                const response = await fetch('/get_history');
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || `HTTP error! status: ${response.status}`);
                }
                const history = await response.json();
                historyListDiv.innerHTML = ''; // Clear existing history
                if (history.length === 0) {
                    historyListDiv.innerHTML = '<p class="text-gray-500">No history available yet.</p>';
                    historyCountSpan.textContent = '(0 entries)';
                    deleteAllHistoryBtn.disabled = true;
                } else {
                    deleteAllHistoryBtn.disabled = false;
                    historyCountSpan.textContent = `(${history.length} entries)`;
                    history.forEach(entry => {
                        const historyItem = document.createElement('div');
                        historyItem.className = 'history-item p-4 mb-2 rounded-md bg-gray-50 hover:bg-gray-100 border border-gray-200 relative';
                        historyItem.innerHTML = `
                            <button class="delete-btn absolute top-2 right-2 text-red-500 hover:text-red-700 focus:outline-none" onclick="deleteEntry(${entry.id})" title="Delete this analysis"><i class="fas fa-trash-alt"></i></button>
                            <h4 class="text-lg font-semibold text-gray-800 cursor-pointer hover:underline" onclick="toggleDetails(${entry.id})">#${entry.id} • ${new Date(entry.timestamp).toLocaleString()}</h4>
                            <div class="history-details" id="details-${entry.id}">
                                <p class="text-gray-600 text-sm"><i class="fas fa-calculator mr-1"></i> Mean: ${entry.mean.toFixed(2)} | STD: ${entry.std_dev.toFixed(2)}</p>
                                <p class="text-gray-600 text-sm"><i class="fas fa-list-ol mr-1"></i> Data points: ${JSON.parse(entry.data).length}</p>
                                <button class="load-entry-btn bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-3 rounded mt-2 text-sm" onclick="loadHistoryEntry(${entry.id})"><i class="fas fa-redo mr-1"></i> Re-analyze</button>
                            </div>
                        `;
                        historyListDiv.appendChild(historyItem);
                    });
                }
            } catch (error) {
                console.error('Error loading history:', error);
                historyListDiv.innerHTML = `<p class="text-red-500">Error loading history: ${error.message}</p>`;
            }
        }

        async function loadHistoryEntry(entryId) {
            showLoading('Loading History Entry...');
            try {
                const response = await fetch(`/get_history`); // Re-fetch history to get data
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const historyEntries = await response.json();
                const entry = historyEntries.find(e => e.id === entryId);
                if (!entry) throw new Error(`Entry ${entryId} not found in history.`);

                const result = {
                    id: entry.id,
                    stats: {
                        mean: entry.mean,
                        std_dev: entry.std_dev,
                        trend: [], // Trend is recalculated on server side, not stored in history (for simplicity)
                        trend_values: []
                    },
                    data: JSON.parse(entry.data)
                };
                // Re-calculate trend on client side to match server-side calculation
                const stats = analyzeDataForTrend(result.data);
                result.stats.trend = stats.trend;
                result.stats.trend_values = stats.trend_values;

                updateUI(result);
                hideLoading();
                scrollToAnalysis(); // Scroll to the analysis results
            } catch (error) {
                hideLoading();
                alert(`Failed to load history entry: ${error.message}`);
                console.error('Error loading history entry:', error);
            }
        }

        // Client-side trend analysis, mirroring server-side logic in analyze_scores function
        function analyzeDataForTrend(data) {
            const y = data; // Data points
            const x = Array.from({ length: y.length }, (_, i) => i); // Indices as x-values
            const trend_coef = np.polyfit(x, y, 1); // Calculate trend line coefficients (slope and intercept)
            const trend_fn = np.poly1d(trend_coef); // Create a polynomial function from the coefficients
            return {
                'trend': list(trend_coef), // Store trend coefficients
                'trend_values': list(trend_fn(x)) // Calculate trend line y-values for each x-value
            };
        }


        function toggleDetails(entryId) {
            const detailsDiv = document.getElementById(`details-${entryId}`);
            detailsDiv.classList.toggle('active');
        }

        async function deleteEntry(entryId) {
            if (!confirm(`Delete analysis #${entryId}? This cannot be undone.`)) return;

            try {
                const response = await fetch('/delete_history', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ entry_id: entryId })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || `Delete failed, HTTP error! status: ${response.status}`);
                }
                await loadHistory();
                resultsDiv.innerHTML = '<p class="text-gray-600">Analysis history updated.</p>'; // Clear results on delete
                currentChart?.destroy(); currentChart = null; // Clear chart
                chartContainer.classList.add('hidden'); // Hide chart container
                resetZoomBtn.disabled = true; // Disable chart controls
                downloadChartBtn.disabled = true;

            } catch (error) {
                alert(`Delete failed: ${error.message}`);
            }
        }

        async function confirmDeleteAll() {
            if (!confirm('Delete ALL analysis history? This cannot be undone!')) return;

            try {
                const response = await fetch('/delete_history', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ delete_all: true })
                });

                if (!response.ok)  {
                    const error = await response.json();
                    throw new Error(error.error || `Delete All failed, HTTP error! status: ${response.status}`);
                }
                await loadHistory();
                resultsDiv.innerHTML = '<p class="text-gray-600">Analysis history cleared.</p>'; // Clear results on delete all
                currentChart?.destroy(); currentChart = null; // Clear chart
                chartContainer.classList.add('hidden'); // Hide chart container
                resetZoomBtn.disabled = true; // Disable chart controls
                downloadChartBtn.disabled = true;
                alert('All history deleted successfully');
            } catch (error) {
                alert(`Delete failed: ${error.message}`);
            }
        }

        document.getElementById('csvFile').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            const errorDiv = document.getElementById('fileError');
            errorDiv.classList.add('hidden');
            if (!file) return; // Exit if no file selected after change event triggered
            showLoading('Analyzing CSV...');

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || `CSV analysis failed, HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                updateUI(result);
                await loadHistory();
                scrollToAnalysis(); // Scroll to the analysis results
            } catch (error) {
                errorDiv.textContent = `Error: ${error.message}`;
                errorDiv.classList.remove('hidden');
            } finally {
                hideLoading();
            }
        });

        function scrollToAnalysis() {
            resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }


        // Initial load
        loadHistory();
        chartContainer.classList.add('hidden'); // Initially hide chart container
    </script>
</body>
</html>
'''


@app.route('/')
def home():
    """
    Route for the home page, rendering the HTML template.
    """
    return render_template_string(HTML_TEMPLATE) # Use HTML_TEMPLATE for full version, or SIMPLIFIED_HTML_TEMPLATE for simplified test

@app.route('/simplified_chart_test') # New route for simplified test page
def simplified_chart_test():
    return render_template_string(SIMPLIFIED_HTML_TEMPLATE)


def calculate_trend_line(data):
    """
    Calculates the linear trend line for the given data.

    Args:
        data (list): List of numerical data points.

    Returns:
        tuple: A tuple containing:
            - list: Trend line coefficients [slope, intercept].
            - list: Trend line values for each data point.
    """
    y = data
    x = np.array(range(len(y)))
    trend_coef = np.polyfit(x, y, 1)
    trend_fn = np.poly1d(trend_coef)
    trend_values = trend_fn(x).tolist()
    return trend_coef.tolist(), trend_values


def analyze_scores_data(data):
    """
    Analyzes the input data to calculate mean, standard deviation, and trend line.

    Args:
        data (list): List of numerical data points.

    Returns:
        dict: Dictionary containing calculated statistics: mean, std_dev, trend coefficients, and trend values.
    """
    df = pd.Series(data)
    trend_coef, trend_values = calculate_trend_line(data)
    return {
        'mean': df.mean(),
        'std_dev': df.std(),
        'trend': trend_coef,
        'trend_values': trend_values
    }


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Route for analyzing data, either from CSV upload or manual input.
    Calculates statistics, stores results in the database, and returns JSON response.
    """
    try:
        data = None # Initialize data to None

        # Handle JSON data input (manual data)
        if request.is_json:
            input_data = request.json.get('data')
            if not input_data or not isinstance(input_data, list) or len(input_data) < 2:
                return jsonify({'error': 'At least 2 valid data points required'}), 400
            try:
                data = [float(x) for x in input_data] # Convert to float and validate number format
            except ValueError:
                return jsonify({'error': 'Invalid data format - must be numbers'}), 400

        # Handle file upload (CSV data)
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if not file.filename.lower().endswith('.csv'):
                return jsonify({'error': 'File must be a CSV'}), 400

            try:
                csv_data = io.StringIO(file.stream.read().decode("UTF8"), newline=None) # Handle file stream directly
                df = pd.read_csv(csv_data)
                numerical_data = df.select_dtypes(include=np.number).values.ravel() # Extract numerical columns
                data = numerical_data[~np.isnan(numerical_data)].tolist() # Remove NaNs and convert to list
                if len(data) < 2:
                    return jsonify({'error': 'At least 2 valid data points required in CSV'}), 400
            except Exception as e:
                return jsonify({'error': f'CSV processing error: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No data provided'}), 400 # No data provided via JSON or file upload

        if data is None: # Double check if data was successfully loaded
            return jsonify({'error': 'Failed to process data'}), 400

        # Common analysis logic
        stats = analyze_scores_data(data)

        # Store results in database
        db = get_db()
        cursor = db.cursor()
        cursor.execute('INSERT INTO scores (data, mean, std_dev) VALUES (?, ?, ?)',
                     (str(data), stats['mean'], stats['std_dev']))
        db.commit()

        return jsonify({
            'stats': stats,
            'id': cursor.lastrowid,
            'data': data
        })

    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}") # Log detailed error on server side
        return jsonify({'error': 'Analysis failed. Please check input data.'}), 500 # Generic error for client


@app.route('/get_history')
def get_history():
    """
    Route to retrieve analysis history from the database.
    Returns a JSON list of history entries, limited by HISTORY_LIMIT.
    """
    try:
        db = get_db()
        results = db.execute(f'''
            SELECT id, data, mean, std_dev, datetime(timestamp) as timestamp
            FROM scores
            ORDER BY timestamp DESC
            LIMIT {HISTORY_LIMIT}
        ''').fetchall()
        return jsonify([dict(row) for row in results])
    except Exception as e:
        app.logger.error(f"History retrieval error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve analysis history'}), 500

@app.route('/delete_history', methods=['POST'])
def delete_history():
    """
    Route to delete analysis history entries.
    Supports deleting a specific entry or all entries.
    """
    try:
        db = get_db()
        if request.json.get('delete_all'):
            # Delete all history
            db.execute('DELETE FROM scores')
            db.commit()
            return jsonify({'success': 'All history deleted'})
        elif request.json.get('entry_id'):
            # Delete specific entry
            entry_id = request.json['entry_id']
            db.execute('DELETE FROM scores WHERE id = ?', (entry_id,))
            db.commit()
            return jsonify({'success': f'Entry {entry_id} deleted'})
        return jsonify({'error': 'Invalid delete request'}), 400
    except Exception as e:
        app.logger.error(f"Delete history error: {str(e)}")
        return jsonify({'error': 'Failed to delete history'}), 500

@app.route('/download_db')
def download_db():
    """
    Route to download the SQLite database file.
    """
    return send_file(
        io.BytesIO(open(DATABASE, 'rb').read()),
        mimetype='application/octet-stream',
        download_name=f'analysis_history_{datetime.utcnow().strftime("%Y%m%d%H%M")}.db',
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)