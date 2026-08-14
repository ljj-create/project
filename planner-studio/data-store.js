const fs = require('fs')
const path = require('path')

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

const DEFAULT_SETTINGS = {
  theme: 'light',
  notify: true,
  autoLaunch: false,
  motto: '今天的努力，是明天的基石',
}
const PRIORITIES = ['high', 'medium', 'low']
const REPEATS = ['none', 'daily', 'workdays', 'weekly']

function defaultData() {
  return {
    futureTasks: [],
    todos: {},
    repeatSkips: {},
    repeatTemplates: {},
    settings: { ...DEFAULT_SETTINGS },
  }
}

function dateKeyOf(d) {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

class DataStore {
  constructor(filePath) {
    this.filePath = filePath
    this.data = this.load()
  }

  load() {
    try {
      const raw = fs.readFileSync(this.filePath, 'utf-8')
      const parsed = JSON.parse(raw)
      const settings = parsed.settings || {}
      return {
        futureTasks: Array.isArray(parsed.futureTasks) ? parsed.futureTasks : [],
        todos: parsed.todos && typeof parsed.todos === 'object' ? parsed.todos : {},
        repeatSkips: parsed.repeatSkips && typeof parsed.repeatSkips === 'object' ? parsed.repeatSkips : {},
        repeatTemplates:
          parsed.repeatTemplates && typeof parsed.repeatTemplates === 'object' ? parsed.repeatTemplates : {},
        settings: { ...DEFAULT_SETTINGS, ...settings },
      }
    } catch {
      return defaultData()
    }
  }

  save() {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true })
    fs.writeFileSync(this.filePath, JSON.stringify(this.data, null, 2))
  }

  getTodos(dateKey) {
    return this.data.todos[dateKey] || []
  }

  getAllTodos() {
    return this.data.todos
  }

  addTodo(dateKey, text, extra = {}) {
    const priority = PRIORITIES.includes(extra.priority) ? extra.priority : 'medium'
    const repeat = REPEATS.includes(extra.repeat) ? extra.repeat : 'none'
    const todo = {
      id: genId(),
      text,
      done: false,
      priority,
      repeat,
      repeatOf: extra.repeatOf || null,
      createdAt: new Date().toISOString(),
    }
    if (todo.repeat !== 'none' && !todo.repeatOf) {
      this.data.repeatTemplates[todo.id] = {
        text: todo.text,
        priority: todo.priority,
        repeat: todo.repeat,
      }
    }
    const list = this.getTodos(dateKey)
    list.push(todo)
    this.data.todos[dateKey] = list
    this.save()
    return list
  }

  toggleTodo(dateKey, id) {
    const todo = this.getTodos(dateKey).find((t) => t.id === id)
    if (todo) {
      todo.done = !todo.done
      todo.doneAt = todo.done ? new Date().toISOString() : null
    }
    this.save()
    return this.getTodos(dateKey)
  }

  removeTodo(dateKey, id) {
    const todo = this.getTodos(dateKey).find((t) => t.id === id)
    if (todo && todo.repeat && todo.repeat !== 'none') {
      const templateId = todo.repeatOf || todo.id
      const skips = this.data.repeatSkips[templateId] || []
      if (!skips.includes(dateKey)) skips.push(dateKey)
      this.data.repeatSkips[templateId] = skips
    }
    this.data.todos[dateKey] = this.getTodos(dateKey).filter((t) => t.id !== id)
    this.save()
    return this.getTodos(dateKey)
  }

  renameTodo(dateKey, id, text) {
    const todo = this.getTodos(dateKey).find((t) => t.id === id)
    if (todo) todo.text = text
    this.save()
    return this.getTodos(dateKey)
  }

  getTasks() {
    return this.data.futureTasks
  }

  addTask(title, dueAt) {
    const task = { id: genId(), title, dueAt, createdAt: new Date().toISOString() }
    this.data.futureTasks.push(task)
    this.save()
    return task
  }

  removeTask(id) {
    this.data.futureTasks = this.data.futureTasks.filter((t) => t.id !== id)
    this.save()
    return this.data.futureTasks
  }

  clearOverdueTasks() {
    const now = Date.now()
    const remaining = this.data.futureTasks.filter((t) => new Date(t.dueAt).getTime() > now)
    this.data.futureTasks = remaining
    this.save()
    return remaining
  }

  getSettings() {
    return this.data.settings
  }

  setSettings(partial) {
    Object.assign(this.data.settings, partial)
    this.save()
    return this.data.settings
  }

  updateRepeat(templateId, repeat) {
    if (!REPEATS.includes(repeat)) return 0
    const tpl = this.data.repeatTemplates[templateId]
    if (tpl) tpl.repeat = repeat
    let n = 0
    for (const dateKey of Object.keys(this.data.todos)) {
      for (const t of this.data.todos[dateKey]) {
        if (t.id === templateId || t.repeatOf === templateId) {
          t.repeat = repeat
          n++
        }
      }
    }
    this.save()
    return n
  }

  deleteRepeatTemplate(templateId) {
    delete this.data.repeatTemplates[templateId]
    delete this.data.repeatSkips[templateId]
    let n = 0
    for (const dateKey of Object.keys(this.data.todos)) {
      const before = this.data.todos[dateKey].length
      this.data.todos[dateKey] = this.data.todos[dateKey].filter(
        (t) => !(t.id === templateId || t.repeatOf === templateId)
      )
      n += before - this.data.todos[dateKey].length
    }
    this.save()
    return n
  }

  shouldRepeat(repeat, date) {
    const day = date.getDay()
    if (repeat === 'daily') return true
    if (repeat === 'workdays') return day >= 1 && day <= 5
    if (repeat === 'weekly') return day === 1
    return false
  }

  applyRepeats(today = new Date()) {
    const todayKey = dateKeyOf(today)
    const todays = this.getTodos(todayKey)
    const hasToday = (templateId) =>
      todays.some((t) => t.repeatOf === templateId || t.id === templateId)
    const seen = new Set()
    let added = 0
    for (let i = 1; i <= 7; i++) {
      const d = new Date(today)
      d.setDate(d.getDate() - i)
      for (const todo of this.getTodos(dateKeyOf(d))) {
        if (!todo.repeat || todo.repeat === 'none') continue
        const templateId = todo.repeatOf || todo.id
        if (seen.has(templateId)) continue
        seen.add(templateId)
        if (hasToday(templateId)) continue
        if (!this.shouldRepeat(todo.repeat, today)) continue
        const skips = this.data.repeatSkips[templateId] || []
        if (skips.includes(todayKey)) continue
        this.addTodo(todayKey, todo.text, {
          priority: todo.priority,
          repeat: todo.repeat,
          repeatOf: templateId,
        })
        added++
      }
    }
    const templates = this.data.repeatTemplates || {}
    for (const [templateId, tpl] of Object.entries(templates)) {
      if (seen.has(templateId)) continue
      seen.add(templateId)
      if (hasToday(templateId)) continue
      if (!this.shouldRepeat(tpl.repeat, today)) continue
      const skips = this.data.repeatSkips[templateId] || []
      if (skips.includes(todayKey)) continue
      this.addTodo(todayKey, tpl.text, {
        priority: tpl.priority,
        repeat: tpl.repeat,
        repeatOf: templateId,
      })
      added++
    }
    return added
  }
}

module.exports = { DataStore, genId, dateKeyOf }
