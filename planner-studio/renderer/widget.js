const api = window.planner

const mottoEl = document.getElementById('motto')
const countdownList = document.getElementById('countdown-list')
const todoList = document.getElementById('todo-list')
const progressBar = document.getElementById('progress-bar')
const progressText = document.getElementById('progress-text')

let tasks = []
let todos = []
let mottoText = ''

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
  countdownList.textContent = ''
  if (tasks.length === 0) {
    const p = document.createElement('p')
    p.className = 'placeholder'
    p.textContent = '没有倒计时任务'
    countdownList.appendChild(p)
    return
  }
  for (const task of tasks) {
    const card = document.createElement('div')
    card.className = 'countdown-card'
    const title = document.createElement('span')
    title.className = 'card-title'
    title.textContent = task.title
    const count = formatCountdown(task)
    const time = document.createElement('span')
    time.className = count.overdue ? 'count-overdue' : 'count'
    time.textContent = count.text
    const due = document.createElement('span')
    due.className = 'card-due'
    due.textContent = new Date(task.dueAt).toLocaleString('zh-CN', {
      month: 'numeric',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
    card.append(title, time, due)
    countdownList.appendChild(card)
  }
}

function renderTodos() {
  todoList.textContent = ''
  const done = todos.filter((t) => t.done).length
  const total = todos.length
  progressBar.style.width = total === 0 ? '0%' : `${Math.round((done / total) * 100)}%`
  progressText.textContent = total === 0 ? '0 / 0' : `${done} / ${total}`
  if (total === 0) {
    const li = document.createElement('li')
    li.className = 'placeholder'
    li.textContent = '今日暂无待办'
    todoList.appendChild(li)
    return
  }
  for (const todo of todos) {
    const li = document.createElement('li')
    li.className = `todo-item p-${todo.priority || 'medium'}`
    if (todo.done) li.classList.add('done')
    const check = document.createElement('input')
    check.type = 'checkbox'
    check.checked = todo.done
    check.addEventListener('change', () => api.todos.toggle(dateKey(), todo.id))
    const label = document.createElement('span')
    label.className = 'todo-text'
    label.textContent = todo.text
    li.append(check, label)
    todoList.appendChild(li)
  }
}

document.addEventListener('contextmenu', (e) => {
  e.preventDefault()
  api.openManager()
})

function renderMotto() {
  mottoEl.textContent = mottoText
}

function startMottoEdit() {
  const input = document.createElement('input')
  input.type = 'text'
  input.className = 'motto-input'
  input.value = mottoText
  input.maxLength = 50
  let done = false
  const finish = (save) => {
    if (done) return
    done = true
    const value = input.value.trim()
    if (save && value && value !== mottoText) {
      api.settings.set({ motto: value })
    }
    input.replaceWith(mottoEl)
    renderMotto()
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
  mottoEl.replaceWith(input)
  input.focus()
  input.select()
}

mottoEl.addEventListener('dblclick', startMottoEdit)

api.onDataChanged(async () => {
  tasks = await api.tasks.list()
  todos = await api.todos.get(dateKey())
  renderTasks()
  renderTodos()
  applySettings()
})

async function applySettings() {
  const settings = await api.settings.get()
  document.body.dataset.theme = settings.theme
  mottoText = settings.motto || ''
  renderMotto()
}

async function init() {
  tasks = await api.tasks.list()
  todos = await api.todos.get(dateKey())
  await applySettings()
  renderTasks()
  renderTodos()
  setInterval(renderTasks, 60000)
}

init()
