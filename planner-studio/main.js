const { app, BrowserWindow, ipcMain, Notification, Tray, Menu, dialog, screen } = require('electron')
const { execFile } = require('child_process')
const fs = require('fs')
const path = require('path')
const { DataStore } = require('./data-store')

let store
let managerWin = null
let widgetWin = null
let tray = null
const notifiedTaskIds = new Set()

const MUTATIONS = [
  'todos:add',
  'todos:toggle',
  'todos:remove',
  'todos:rename',
  'todos:updateRepeat',
  'todos:deleteRepeat',
  'tasks:add',
  'tasks:remove',
  'tasks:clearOverdue',
  'settings:set',
  'app:autoLaunch',
  'backup:import',
]

function broadcast() {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('data-changed')
  }
}

function attachToDesktop(win) {
  try {
    const hwnd = win.getNativeWindowHandle().readUInt32LE(0)
    const script = path.join(__dirname, 'assets', 'attach-desktop.ps1')
    const ps = process.env.SystemRoot + '\\System32\\WindowsPowerShell\\v1.0\\powershell.exe'
    execFile(ps, ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script, String(hwnd)], (err) => {
      if (err) console.error('attach to desktop failed:', err.message)
    })
  } catch (e) {
    console.error('attach to desktop failed:', e.message)
  }
}

const WIDGET_W = 340
const WIDGET_H = 560
const isAutoStart = process.argv.includes('--autostart')

function widgetTopRightBounds() {
  const wa = screen.getPrimaryDisplay().workArea
  const margin = 16
  return { x: wa.x + wa.width - WIDGET_W - margin, y: wa.y + margin }
}

function createWidget() {
  const win = new BrowserWindow({
    width: WIDGET_W,
    height: WIDGET_H,
    frame: false,
    transparent: true,
    alwaysOnTop: false,
    skipTaskbar: true,
    resizable: false,
    title: 'Planner Studio',
    ...(isAutoStart ? widgetTopRightBounds() : {}),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })
  widgetWin = win
  win.loadFile(path.join(__dirname, 'renderer', 'widget.html'))
  win.webContents.once('did-finish-load', () => attachToDesktop(win))
  win.on('closed', () => app.quit())
}

function openManager() {
  if (managerWin && !managerWin.isDestroyed()) {
    managerWin.show()
    managerWin.focus()
    return
  }
  managerWin = new BrowserWindow({
    width: 720,
    height: 860,
    title: 'Planner Studio - 管理',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })
  managerWin.loadFile(path.join(__dirname, 'renderer', 'manager.html'))
  managerWin.on('closed', () => (managerWin = null))
}

function createTray() {
  tray = new Tray(path.join(__dirname, 'assets', 'tray.png'))
  tray.setToolTip('Planner Studio')
  tray.setContextMenu(
    Menu.buildFromTemplate([
      {
        label: '显示挂件',
        click: () => {
          widgetWin.show()
          widgetWin.focus()
        },
      },
      {
        label: '隐藏挂件',
        click: () => widgetWin.hide(),
      },
      { type: 'separator' },
      {
        label: '退出',
        click: () => app.quit(),
      },
    ])
  )
}

function checkDueTasks() {
  const now = Date.now()
  const settings = store.getSettings()
  for (const task of store.getTasks()) {
    if (new Date(task.dueAt).getTime() <= now && !notifiedTaskIds.has(task.id)) {
      notifiedTaskIds.add(task.id)
      if (settings.notify && Notification.isSupported()) {
        new Notification({ title: '任务到期', body: task.title }).show()
      }
    }
  }
}

function checkRepeats() {
  if (store.applyRepeats() > 0) broadcast()
}

function registerIpc() {
  const handle = (channel, fn) =>
    ipcMain.handle(channel, (_e, ...args) => {
      const result = fn(...args)
      if (MUTATIONS.includes(channel)) broadcast()
      return result
    })
  handle('todos:get', (dateKey) => store.getTodos(dateKey))
  handle('todos:all', () => store.getAllTodos())
  handle('todos:add', (dateKey, text, extra) => store.addTodo(dateKey, text, extra))
  handle('todos:toggle', (dateKey, id) => store.toggleTodo(dateKey, id))
  handle('todos:remove', (dateKey, id) => store.removeTodo(dateKey, id))
  handle('todos:rename', (dateKey, id, text) => store.renameTodo(dateKey, id, text))
  handle('todos:updateRepeat', (templateId, repeat) => store.updateRepeat(templateId, repeat))
  handle('todos:deleteRepeat', (templateId) => store.deleteRepeatTemplate(templateId))
  handle('tasks:list', () => store.getTasks())
  handle('tasks:add', (title, dueAt) => store.addTask(title, dueAt))
  handle('tasks:remove', (id) => store.removeTask(id))
  handle('tasks:clearOverdue', () => store.clearOverdueTasks())
  handle('settings:get', () => store.getSettings())
  handle('settings:set', (partial) => store.setSettings(partial))
  handle('app:autoLaunch', (enabled) => {
    app.setLoginItemSettings({
      openAtLogin: !!enabled,
      args: ['--autostart'],
    })
    return store.setSettings({ autoLaunch: !!enabled })
  })
  handle('backup:export', async () => {
    const dateStr = new Date().toISOString().slice(0, 10)
    const { canceled, filePath } = await dialog.showSaveDialog(managerWin, {
      title: '导出备份',
      defaultPath: `planner-backup-${dateStr}.json`,
      filters: [{ name: 'JSON', extensions: ['json'] }],
    })
    if (canceled || !filePath) return { ok: false }
    fs.writeFileSync(filePath, JSON.stringify(store.data, null, 2))
    return { ok: true, path: filePath }
  })
  handle('backup:import', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog(managerWin, {
      title: '导入备份',
      filters: [{ name: 'JSON', extensions: ['json'] }],
      properties: ['openFile'],
    })
    if (canceled || !filePaths[0]) return { ok: false, reason: '已取消' }
    try {
      const parsed = JSON.parse(fs.readFileSync(filePaths[0], 'utf-8'))
      if (!Array.isArray(parsed.futureTasks) || typeof parsed.todos !== 'object') {
        return { ok: false, reason: '备份格式不正确' }
      }
      store.data = {
        futureTasks: parsed.futureTasks,
        todos: parsed.todos,
        settings: store.data.settings,
      }
      store.save()
      return { ok: true }
    } catch {
      return { ok: false, reason: '文件读取失败' }
    }
  })
  ipcMain.on('window:close', () => widgetWin.hide())
  ipcMain.on('manager:open', () => openManager())
}

app.whenReady().then(() => {
  app.setAppUserModelId('com.planner.studio')
  store = new DataStore(path.join(app.getPath('userData'), 'data.json'))
  registerIpc()
  createWidget()
  createTray()
  checkDueTasks()
  checkRepeats()
  setInterval(() => {
    checkDueTasks()
    checkRepeats()
  }, 30000)
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
