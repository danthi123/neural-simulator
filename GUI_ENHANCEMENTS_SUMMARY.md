# GUI Enhancements - Performance Testing & System Logs

## Overview
Added two new collapsible GUI sections to the neural simulator interface for better user control over performance testing and system monitoring.

## Changes Made

### 1. LogCapture Class (Lines 453-519)
**Purpose**: Thread-safe log capture system for displaying console output in the GUI.

**Features**:
- Captures all `print()` statements and stderr output
- Thread-safe buffer with configurable maximum lines (default: 5000)
- Search functionality with case-insensitive matching
- Maintains original stdout/stderr while capturing
- Can be started/stopped dynamically

**Key Methods**:
- `start_capture()` - Begin capturing console output
- `stop_capture()` - Restore original stdout/stderr
- `get_logs()` - Retrieve all captured log lines
- `clear()` - Clear the log buffer
- `search(query, case_sensitive)` - Find matching lines

### 2. Performance Testing & Optimization Section (Lines 6877-6900)
**Location**: After "Visual Settings & Filters" section in GUI

**Features**:
- **Run Benchmark Suite**: Executes `benchmark.py` in background thread
  - Shows status and results in GUI
  - Outputs to `benchmark_results.json`
  - 300-second timeout for safety
  
- **Run Auto-Tuning (Optimize Drive Scales)**: Runs the auto-tuning workflow
  - Optimizes external drive scales for different model/profile combinations
  - "Quick" checkbox for faster testing (off by default)
  - Quick mode: Fewer configurations, faster completion
  - Full mode: Comprehensive tuning across all combinations
  
- **Reload Auto-Tuned Overrides**: Reloads from `auto_tuned_overrides.json`
  - Forces fresh read from disk
  - Displays count of available combinations
  - Auto-applied on next "Apply Changes & Reset"

**UI Elements**:
- Status text field showing current operation state
- Multi-line results display (80px height, read-only)
- Full-width buttons for each operation

### 3. System Logs Section (Lines 6904-6920)
**Location**: After "Performance Testing & Optimization" section in GUI

**Features**:
- **Search Functionality**:
  - Real-time search with Previous/Next navigation
  - Match counter display (e.g., "3 / 15 matches")
  - Context-aware highlighting (shows 5 lines before/after match)
  - Search buttons enabled only when matches exist
  
- **Auto-scroll Toggle**:
  - When enabled, displays last 20 log lines
  - Updated every main loop iteration
  - Can be disabled for manual navigation
  
- **Clear Logs**: Wipes the entire log buffer
  
- **Export Logs**: Saves to timestamped file
  - Format: `simulation_logs_YYYYMMDD_HHMMSS.txt`
  - Exports complete log buffer

**UI Elements**:
- Search input field (220px width)
- Previous/Next navigation buttons (70px each)
- Match counter text
- Auto-scroll checkbox
- Clear/Export buttons (100px each)
- Multi-line log display (200px height, read-only)

### 4. Handler Functions (Lines 6560-6728)

#### Performance Testing Handlers:
- `handle_run_benchmark_click()` - Spawns thread to run benchmark.py via subprocess
- `handle_run_optimization_click()` - Placeholder with informational message
- `handle_reload_overrides_click()` - Forces reload of AUTO_TUNED_OVERRIDES

#### Log System Handlers:
- `handle_log_search_change()` - Updates search results and enables/disables navigation
- `handle_log_search_prev()` - Cycles to previous match with wraparound
- `handle_log_search_next()` - Cycles to next match with wraparound
- `_update_log_display_with_highlight()` - Shows context with ">>>" marker for matched line
- `handle_clear_logs_click()` - Clears buffer and display
- `handle_export_logs_click()` - Writes logs to timestamped file

### 5. Main Loop Integration (Lines 7460-7472, 7808-7812)

**Initialization**:
- LogCapture instance created in `main()` before GUI setup
- Capture started immediately to catch all subsequent output
- Reference stored on handler function for global access

**Update Loop**:
- Checks if auto-scroll is enabled every iteration
- Updates display with last 20 lines if logs have changed
- Only updates when display content differs (avoids flicker)

## Design Decisions

### Thread Safety
- LogCapture uses `threading.Lock()` for buffer access
- Benchmark runs in daemon thread to avoid blocking UI
- All DPG updates checked with `dpg.is_dearpygui_running()`

### Performance
- Log display limited to 20 lines for rendering efficiency
- Search is case-insensitive by default for better UX
- Display only updates when content changes
- Main loop integration adds minimal overhead

### Layout Consistency
- Uses existing `label_col_width` (320px) for consistency
- Matches spacing patterns from other sections (3-5px spacers)
- Buttons use full width (`width=-1`) to fill available space
- Multi-line text fields use readonly mode for display-only content

### Error Handling
- All handlers wrapped in try-except blocks
- Subprocess timeout prevents hanging on benchmark failures
- Graceful degradation if log_capture reference missing
- Status bar feedback for all user actions

## File Paths
- Logs exported to: `simulation_logs_YYYYMMDD_HHMMSS.txt`
- Benchmark results: `benchmark_results.json` (from benchmark.py)
- Auto-tuned overrides: `simulation_profiles/auto_tuned_overrides.json`

## Future Enhancements (Phase B)
1. **Model Optimization Implementation**:
   - Automatic drive scale tuning workflow
   - Parameter sweeps for rate targets
   - Integration with existing auto-tuning infrastructure

2. **Visualization Performance Test** (deferred):
   - VizPerformanceBenchmark class
   - Automatic optimal update interval detection
   - Profile persistence and auto-application

3. **Enhanced Log Features**:
   - Log level filtering (INFO, WARNING, ERROR)
   - Timestamps on each log entry
   - Persistent log files across sessions

## Recent Fixes (2025-12-01)

### Benchmark OOM Prevention
- Added memory estimation before each benchmark configuration
- Pre-checks available GPU VRAM (uses 80% of free memory as safe limit)
- Skips configurations that would exceed available memory
- Shows clear messages: "Estimated memory needed: X.XXgb" and "GPU free memory: Y.YYgb"
- Prevents mid-benchmark crashes from out-of-memory errors

### Real-Time Log Output
- Added `sys.stdout.flush()` after all print statements in benchmark.py
- Logs now appear in System Logs section as benchmark runs (not just at completion)
- Progress updates visible every 100 steps
- User can see which configuration is running in real-time

### Auto-Tuning Integration
- "Run Model Optimization" button now functional (was placeholder)
- Renamed to "Run Auto-Tuning (Optimize Drive Scales)" for clarity
- Added "Quick" checkbox next to button (off by default)
  - Quick mode: Passes `--quick` flag, tests fewer configurations
  - Full mode: Comprehensive tuning across all model/profile combinations
- Spawns subprocess with `--auto-tune` flag (and `--quick` if checked)
- Status text shows which mode is running: "(quick mode)" or "(full mode)"
- Shows progress in System Logs section
- Displays completion status and tuned combination count
- Includes 30-minute timeout protection
- Reminds user to click "Reload Auto-Tuned Overrides" after completion

See `GUI_FIXES_SUMMARY.md` for detailed information on these fixes.

## Testing Checklist
- [x] Python compilation check passes
- [x] Benchmark checks VRAM before attempting configs
- [x] Benchmark shows real-time logs during execution
- [x] Auto-tuning button spawns actual optimization workflow
- [ ] GUI sections render without layout overflow
- [ ] Log search finds matches and navigation works
- [ ] Auto-scroll updates display in real-time
- [ ] Export logs creates valid timestamped file
- [ ] Reload overrides updates counter correctly
- [ ] All buttons and inputs maintain styling consistency

## Notes
- Log capture begins immediately when application starts
- Search is performed on complete log buffer (up to 5000 lines)
- Benchmark subprocess inherits current Python interpreter path
- All new GUI elements use tags for programmatic access
- Status bar provides feedback for all major operations
