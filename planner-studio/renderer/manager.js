const api = window.planner

const todayEl = document.getElementById('today')
const taskForm = document.getElementById('task-form')
const taskTitle = document.getElementById('task-title')
const taskDue = document.getElementById('task-due')
const taskList = document.getElementById('task-list')
const todoForm = document.getElementById('todo-form')
const todoInput = document.getElementById('todo-input')
const todoPriority = document.getElementById('todo-priority')
const todoRepeat = document.getElementById('todo-repeat')
const todoList = document.getElementById('todo-list')
const todayProgress = document.getElementById('today-progress')
const clearOverdueBtn = document.getElementById('clear-overdue-btn')
const statToday = document.getElementById('stat-today')
const statWeek = document.getElementById('stat-week')
const statStreak = document.getElementById('stat-streak')
const weekChart = document.getElementById('week-chart')
const settingTheme = document.getElementById('setting-theme')
const settingNotify = document.getElementById('setting-notify')
const settingAutoLaunch = document.getElementById('setting-autolaunch')
const backupExportBtn = document.getElementById('backup-export-btn')
const backupImportBtn = document.getElementById('backup-import-btn')
const calPrev = document.getElementById('cal-prev')
const calNext = document.getElementById('cal-next')
const calTitle = document.getElementById('cal-title')
const calGrid = document.getElementById('cal-grid')
const calDayTitle = document.getElementById('cal-day-title')
const calDayTodos = document.getElementById('cal-day-todos')

let tasks = []
let todos = []
let allTodos = {}
const now = new Date()
let calYear = now.getFullYear()
let calMonth = now.getMonth()
let selectedKey = dateKey(now)

function dateKey(d = new Date()) {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

function formatCountdown(task) {
  const diff = new Date(task.dueAt).getTime() - Date.now()
  if (diff <= 0) return { text: '已过期', overdue: true }
  const days = Math.floor(diff / 86400000)
  const hours = Math.floor(diff / 3600000) % 24
  const mins = Math.floor(diff / 60000) % 60
  if (days > 0) return { text: `还剩 ${days} 天 ${hours} 小时`, overdue: false }
  if (hours > 0) return { text: `还剩 ${hours} 小时 ${mins} 分钟`, overdue: false }
  return { text: `还剩 ${mins} 分钟`, overdue: false }
}

function renderTasks() {
  taskList.textContent = ''
  if (tasks.length === 0) {
    const p = document.createElement('p')
    p.className = 'placeholder'
    p.textContent = '还没有未来任务'
    taskList.appendChild(p)
    return
  }
  for (const task of tasks) {
    const card = document.createElement('div')
    card.className = 'task-card'
    const title = document.createElement('span')
    title.className = 'card-title'
    title.textContent = task.title
    const count = formatCountdown(task)
    const time = document.createElement('span')
    time.className = count.overdue ? 'count overdue' : 'count'
    time.textContent = count.text
    const due = document.createElement('span')
    due.className = 'card-due'
    due.textContent = new Date(task.dueAt).toLocaleString('zh-CN', {
      month: 'numeric',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
    const del = document.createElement('button')
    del.className = 'icon-btn'
    del.textContent = '×'
    del.title = '删除'
    del.addEventListener('click', () => api.tasks.remove(task.id))
    card.append(title, time, due, del)
    taskList.appendChild(card)
  }
}

const PRIORITY_CLASS = { high: 'p-high', medium: 'p-medium', low: 'p-low' }

function renderTodos() {
  todoList.textContent = ''
  const sorted = [...todos].sort((a, b) => {
    const rank = { high: 0, medium: 1, low: 2 }
    return rank[a.priority || 'medium'] - rank[b.priority || 'medium']
  })
  const done = sorted.filter((t) => t.done).length
  const total = sorted.length
  todayProgress.textContent = total === 0 ? '' : `已完成 ${done} / ${total}`
  if (total === 0) {
    const li = document.createElement('li')
    li.className = 'placeholder'
    li.textContent = '今天还没有安排'
    todoList.appendChild(li)
    return
  }
  for (const todo of sorted) {
    const li = document.createElement('li')
    li.className = `todo-item ${PRIORITY_CLASS[todo.priority || 'medium']}`
    if (todo.done) li.classList.add('done')
    const check = document.createElement('input')
    check.type = 'checkbox'
    check.checked = todo.done
    check.addEventListener('change', () => api.todos.toggle(dateKey(), todo.id))
    const label = document.createElement('span')
    label.className = 'todo-text'
    label.textContent = todo.text
    label.title = '双击编辑'
    label.addEventListener('dblclick', () => startEdit(label, dateKey(), todo.id, todo.text))
    if (todo.repeat && todo.repeat !== 'none') {
      const tag = document.createElement('button')
      tag.className = 'repeat-tag'
      tag.textContent = { daily: '每日', workdays: '工作日', weekly: '每周一' }[todo.repeat]
      tag.title = '修改重复规则或永久删除'
      tag.addEventListener('click', (e) => {
        e.stopPropagation()
        showRepeatMenu(todo, tag)
      })
      label.appendChild(document.createTextNode(' '))
      label.appendChild(tag)
    }
    const del = document.createElement('button')
    del.className = 'icon-btn'
    del.textContent = '×'
    del.title = '删除'
    del.addEventListener('click', () => api.todos.remove(dateKey(), todo.id))
    li.append(check, label, del)
    todoList.appendChild(li)
  }
}

function renderStats() {
  const days = []
  for (let i = 6; i >= 0; i--) {
    const d = new Date()
    d.setDate(d.getDate() - i)
    const key = dateKey(d)
    const list = allTodos[key] || []
    const done = list.filter((t) => t.done).length
    days.push({ key, total: list.length, done, weekday: d.toLocaleDateString('zh-CN', { weekday: 'short' }) })
  }
  statToday.textContent = days[6].total === 0 ? '-' : `${days[6].done} / ${days[6].total}`

  let weekTotal = 0
  let weekDone = 0
  for (const d of days) {
    weekTotal += d.total
    weekDone += d.done
  }
  statWeek.textContent = weekTotal === 0 ? '-' : `${Math.round((weekDone / weekTotal) * 100)}%`

  let streak = 0
  const d = new Date()
  while (true) {
    const list = (allTodos[dateKey(d)] || []).filter((t) => t.done)
    if (list.length === 0) break
    streak++
    d.setDate(d.getDate() - 1)
  }
  statStreak.textContent = streak

  weekChart.textContent = ''
  for (const day of days) {
    const col = document.createElement('div')
    col.className = 'chart-col'
    const bar = document.createElement('div')
    bar.className = 'chart-bar'
    const pct = day.total === 0 ? 0 : Math.round((day.done / day.total) * 100)
    bar.style.height = `${Math.max(pct, 2)}%`
    bar.title = `${day.key} 完成 ${day.done}/${day.total}`
    const label = document.createElement('span')
    label.className = 'chart-label'
    label.textContent = day.weekday
    col.append(bar, label)
    weekChart.appendChild(col)
  }
}

async function applyTheme() {
  const settings = await api.settings.get()
  document.body.dataset.theme = settings.theme
  return settings
}

async function syncSettings() {
  const settings = await applyTheme()
  settingTheme.value = settings.theme
  settingNotify.checked = settings.notify
  settingAutoLaunch.checked = settings.autoLaunch
}

function startEdit(label, key, id, text) {
  const input = document.createElement('input')
  input.type = 'text'
  input.className = 'edit-input'
  input.value = text
  let done = false
  const finish = (save) => {
    if (done) return
    done = true
    const value = input.value.trim()
    if (save && value && value !== text) {
      api.todos.rename(key, id, value)
    }
    input.replaceWith(label)
    label.textContent = text
  }
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      finish(true)
      input.blur()
    } else if (e.key === 'Escape') {
      finish(false)
      input.blur()
    }
  })
  input.addEventListener('blur', () => finish(false))
  label.replaceWith(input)
  input.focus()
  input.select()
}

function showRepeatMenu(todo, anchor) {
  const templateId = todo.repeatOf || todo.id
  document.querySelector('.repeat-menu')?.remove()
  const menu = document.createElement('div')
  menu.className = 'repeat-menu'
  const title = document.createElement('div')
  title.className = 'menu-title'
  title.textContent = '修改重复规则'
  menu.appendChild(title)
  for (const [val, label] of [
    ['daily', '每天'],
    ['workdays', '工作日'],
    ['weekly', '每周一'],
    ['none', '不重复'],
  ]) {
    const b = document.createElement('button')
    b.textContent = label + (todo.repeat === val ? ' ✓' : '')
    b.addEventListener('click', async () => {
      menu.remove()
      await api.todos.updateRepeat(templateId, val)
    })
    menu.appendChild(b)
  }
  const sep = document.createElement('div')
  sep.className = 'menu-sep'
  const del = document.createElement('button')
  del.className = 'danger'
  del.textContent = '永久删除此任务'
  let armed = false
  del.addEventListener('click', async () => {
    if (!armed) {
      armed = true
      del.textContent = '再次点击确认永久删除'
      return
    }
    menu.remove()
    await api.todos.deleteRepeat(templateId)
  })
  menu.append(sep, del)
  document.body.appendChild(menu)
  const r = anchor.getBoundingClientRect()
  menu.style.left = `${Math.min(r.left, window.innerWidth - 160)}px`
  menu.style.top = `${r.bottom + 4}px`
  setTimeout(() => {
    document.addEventListener(
      'click',
      (e) => {
        if (!menu.contains(e.target)) menu.remove()
      },
      { once: true }
    )
  }, 0)
}

function renderCalendar() {
  calTitle.textContent = `${calYear}年${calMonth + 1}月`
  calGrid.textContent = ''
  const first = new Date(calYear, calMonth, 1)
  const startDay = first.getDay()
  const todayKey = dateKey(new Date())
  for (let i = 0; i < 42; i++) {
    const d = new Date(calYear, calMonth, 1 - startDay + i)
    const key = dateKey(d)
    const cell = document.createElement('div')
    cell.className = 'cal-cell'
    cell.textContent = d.getDate()
    if (d.getMonth() !== calMonth) cell.classList.add('dim')
    if (key === todayKey) cell.classList.add('today')
    if (key === selectedKey) cell.classList.add('selected')
    const hasTodo = (allTodos[key] || []).length > 0
    const hasTask = tasks.some((t) => dateKey(new Date(t.dueAt)) === key)
    if (hasTodo || hasTask) {
      const dot = document.createElement('span')
      dot.className = hasTask ? 'dot task-dot' : 'dot'
      cell.appendChild(dot)
    }
    cell.addEventListener('click', () => {
      selectedKey = key
      renderCalendar()
      renderSelectedDay()
    })
    calGrid.appendChild(cell)
  }
  renderSelectedDay()
}

function renderSelectedDay() {
  calDayTitle.textContent = selectedKey === dateKey(new Date()) ? '今日待办' : `${selectedKey} 待办`
  calDayTodos.textContent = ''
  const dayTodos = (allTodos[selectedKey] || []).sort((a, b) => {
    const rank = { high: 0, medium: 1, low: 2 }
    return rank[a.priority || 'medium'] - rank[b.priority || 'medium']
  })
  const dayTasks = tasks.filter((t) => dateKey(new Date(t.dueAt)) === selectedKey)
  for (const task of dayTasks) {
    const li = document.createElement('li')
    li.className = 'cal-task'
    li.textContent = `⏰ ${task.title}`
    calDayTodos.appendChild(li)
  }
  if (dayTodos.length === 0 && dayTasks.length === 0) {
    const li = document.createElement('li')
    li.className = 'placeholder'
    li.textContent = '这一天没有安排'
    calDayTodos.appendChild(li)
    return
  }
  for (const todo of dayTodos) {
    const li = document.createElement('li')
    li.className = `todo-item ${PRIORITY_CLASS[todo.priority || 'medium']}`
    if (todo.done) li.classList.add('done')
    const check = document.createElement('input')
    check.type = 'checkbox'
    check.checked = todo.done
    check.addEventListener('change', () => api.todos.toggle(selectedKey, todo.id))
    const label = document.createElement('span')
    label.className = 'todo-text'
    label.textContent = todo.text
    const del = document.createElement('button')
    del.className = 'icon-btn'
    del.textContent = '×'
    del.title = '删除'
    del.addEventListener('click', () => api.todos.remove(selectedKey, todo.id))
    li.append(check, label, del)
    calDayTodos.appendChild(li)
  }
}

calPrev.addEventListener('click', () => {
  calMonth--
  if (calMonth < 0) {
    calMonth = 11
    calYear--
  }
  renderCalendar()
})

calNext.addEventListener('click', () => {
  calMonth++
  if (calMonth > 11) {
    calMonth = 0
    calYear++
  }
  renderCalendar()
})

backupExportBtn.addEventListener('click', async () => {
  const result = await api.backup.export()
  alert(result.ok ? `备份已保存到：${result.path}` : '已取消导出')
})

backupImportBtn.addEventListener('click', async () => {
  const result = await api.backup.import()
  alert(result.ok ? '备份导入成功' : `导入失败：${result.reason || '已取消'}`)
})

taskForm.addEventListener('submit', async (e) => {
  e.preventDefault()
  await api.tasks.add(taskTitle.value.trim(), taskDue.value)
  taskTitle.value = ''
  taskDue.value = ''
})

todoForm.addEventListener('submit', async (e) => {
  e.preventDefault()
  await api.todos.add(dateKey(), todoInput.value.trim(), {
    priority: todoPriority.value,
    repeat: todoRepeat.value,
  })
  todoInput.value = ''
  todoPriority.value = 'medium'
  todoRepeat.value = 'none'
})

clearOverdueBtn.addEventListener('click', () => api.tasks.clearOverdue())

settingTheme.addEventListener('change', () => api.settings.set({ theme: settingTheme.value }))
settingNotify.addEventListener('change', () => api.settings.set({ notify: settingNotify.checked }))
settingAutoLaunch.addEventListener('change', () => api.autoLaunch(settingAutoLaunch.checked))

api.onDataChanged(async () => {
  tasks = await api.tasks.list()
  todos = await api.todos.get(dateKey())
  allTodos = await api.todos.all()
  renderTasks()
  renderTodos()
  renderStats()
  renderCalendar()
  syncSettings()
})

async function init() {
  todayEl.textContent = new Date().toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'long',
  })
  tasks = await api.tasks.list()
  todos = await api.todos.get(dateKey())
  allTodos = await api.todos.all()
  renderTasks()
  renderTodos()
  renderStats()
  renderCalendar()
  syncSettings()
  setInterval(() => {
    renderTasks()
    renderStats()
  }, 60000)
}

init()
