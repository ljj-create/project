const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { DataStore, genId, dateKeyOf } = require('../data-store')

function tmpStore() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'planner-test-'))
  return { store: new DataStore(path.join(dir, 'data.json')), dir }
}

test('genId 生成唯一 id', () => {
  const ids = new Set(Array.from({ length: 100 }, genId))
  assert.equal(ids.size, 100)
})

test('首次加载返回默认结构', () => {
  const { store, dir } = tmpStore()
  assert.deepEqual(store.getTasks(), [])
  assert.deepEqual(store.getTodos('2026-08-12'), [])
  fs.rmSync(dir, { recursive: true, force: true })
})

test('addTodo 追加待办并持久化', () => {
  const { store, dir } = tmpStore()
  store.addTodo('2026-08-12', '晨跑')
  store.addTodo('2026-08-12', '写报告')

  const reloaded = new DataStore(path.join(dir, 'data.json'))
  const list = reloaded.getTodos('2026-08-12')
  assert.equal(list.length, 2)
  assert.equal(list[0].text, '晨跑')
  assert.equal(list[0].done, false)
  assert.ok(list[0].id)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('toggleTodo 切换完成状态并持久化', () => {
  const { store, dir } = tmpStore()
  const todo = store.addTodo('2026-08-12', '晨跑').at(-1)
  store.toggleTodo('2026-08-12', todo.id)

  const reloaded = new DataStore(path.join(dir, 'data.json'))
  assert.equal(reloaded.getTodos('2026-08-12')[0].done, true)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('removeTodo 删除指定待办', () => {
  const { store, dir } = tmpStore()
  const a = store.addTodo('2026-08-12', 'a').at(-1)
  const b = store.addTodo('2026-08-12', 'b').at(-1)
  store.removeTodo('2026-08-12', a.id)

  const list = store.getTodos('2026-08-12')
  assert.equal(list.length, 1)
  assert.equal(list[0].id, b.id)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('不同日期的待办互不干扰', () => {
  const { store, dir } = tmpStore()
  store.addTodo('2026-08-12', '今天的事')
  store.addTodo('2026-08-13', '明天的事')
  assert.equal(store.getTodos('2026-08-12').length, 1)
  assert.equal(store.getTodos('2026-08-13').length, 1)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('addTask 添加未来任务并持久化', () => {
  const { store, dir } = tmpStore()
  store.addTask('交季度报告', '2026-09-30T18:00:00')

  const reloaded = new DataStore(path.join(dir, 'data.json'))
  const task = reloaded.getTasks()[0]
  assert.equal(task.title, '交季度报告')
  assert.equal(task.dueAt, '2026-09-30T18:00:00')
  fs.rmSync(dir, { recursive: true, force: true })
})

test('removeTask 删除指定任务', () => {
  const { store, dir } = tmpStore()
  const t1 = store.addTask('A', '2026-09-01T00:00:00')
  store.addTask('B', '2026-09-02T00:00:00')
  store.removeTask(t1.id)
  assert.equal(store.getTasks().length, 1)
  assert.equal(store.getTasks()[0].title, 'B')
  fs.rmSync(dir, { recursive: true, force: true })
})

test('clearOverdueTasks 只清除已过期的任务', () => {
  const { store, dir } = tmpStore()
  store.addTask('已过期任务', '2000-01-01T00:00:00')
  store.addTask('未来任务', '2999-01-01T00:00:00')
  const remaining = store.clearOverdueTasks()
  assert.equal(remaining.length, 1)
  assert.equal(remaining[0].title, '未来任务')

  const reloaded = new DataStore(path.join(dir, 'data.json'))
  assert.equal(reloaded.getTasks().length, 1)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('addTodo 支持优先级和重复设置', () => {
  const { store, dir } = tmpStore()
  store.addTodo('2026-08-12', '重要', { priority: 'high', repeat: 'daily' })
  const todo = store.getTodos('2026-08-12')[0]
  assert.equal(todo.priority, 'high')
  assert.equal(todo.repeat, 'daily')
  assert.equal(todo.repeatOf, null)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('addTodo 非法优先级/重复值回退默认', () => {
  const { store, dir } = tmpStore()
  store.addTodo('2026-08-12', 'x', { priority: 'urgent', repeat: 'hourly' })
  const todo = store.getTodos('2026-08-12')[0]
  assert.equal(todo.priority, 'medium')
  assert.equal(todo.repeat, 'none')
  fs.rmSync(dir, { recursive: true, force: true })
})

test('toggleTodo 记录完成时间，取消完成清空', () => {
  const { store, dir } = tmpStore()
  const todo = store.addTodo('2026-08-12', '晨跑').at(-1)
  store.toggleTodo('2026-08-12', todo.id)
  assert.ok(store.getTodos('2026-08-12')[0].doneAt)
  store.toggleTodo('2026-08-12', todo.id)
  assert.equal(store.getTodos('2026-08-12')[0].doneAt, null)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('设置读写与持久化', () => {
  const { store, dir } = tmpStore()
  assert.equal(store.getSettings().theme, 'light')
  store.setSettings({ theme: 'dark' })
  const reloaded = new DataStore(path.join(dir, 'data.json'))
  assert.equal(reloaded.getSettings().theme, 'dark')
  fs.rmSync(dir, { recursive: true, force: true })
})

test('shouldRepeat 判断', () => {
  const { store, dir } = tmpStore()
  const mon = new Date('2026-08-10T12:00:00')
  const sat = new Date('2026-08-15T12:00:00')
  assert.equal(store.shouldRepeat('daily', sat), true)
  assert.equal(store.shouldRepeat('workdays', mon), true)
  assert.equal(store.shouldRepeat('workdays', sat), false)
  assert.equal(store.shouldRepeat('weekly', mon), true)
  assert.equal(store.shouldRepeat('weekly', sat), false)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('applyRepeats 每日任务自动生成且不重复', () => {
  const { store, dir } = tmpStore()
  const tuesday = new Date('2026-08-11T12:00:00')
  store.addTodo(dateKeyOf(tuesday), '喝水', { repeat: 'daily' })
  const wednesday = new Date('2026-08-12T12:00:00')
  assert.equal(store.applyRepeats(wednesday), 1)
  const list = store.getTodos(dateKeyOf(wednesday))
  assert.equal(list.length, 1)
  assert.equal(list[0].text, '喝水')
  assert.equal(list[0].repeatOf, store.getTodos(dateKeyOf(tuesday))[0].id)
  assert.equal(store.applyRepeats(wednesday), 0)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('applyRepeats 周末不生成工作日任务', () => {
  const { store, dir } = tmpStore()
  const friday = new Date('2026-08-14T12:00:00')
  store.addTodo(dateKeyOf(friday), '打卡', { repeat: 'workdays' })
  const saturday = new Date('2026-08-15T12:00:00')
  assert.equal(store.applyRepeats(saturday), 0)
  const monday = new Date('2026-08-17T12:00:00')
  assert.equal(store.applyRepeats(monday), 1)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('renameTodo 修改待办文字并持久化', () => {
  const { store, dir } = tmpStore()
  const todo = store.addTodo('2026-08-12', '原文字').at(-1)
  store.renameTodo('2026-08-12', todo.id, '新文字')
  const reloaded = new DataStore(path.join(dir, 'data.json'))
  assert.equal(reloaded.getTodos('2026-08-12')[0].text, '新文字')
  fs.rmSync(dir, { recursive: true, force: true })
})

test('删除某天的重复实例后当天不生成，次日恢复', () => {
  const { store, dir } = tmpStore()
  const tue = new Date('2026-08-11T12:00:00')
  store.addTodo(dateKeyOf(tue), '喝水', { repeat: 'daily' })
  const wed = new Date('2026-08-12T12:00:00')
  assert.equal(store.applyRepeats(wed), 1)
  const todayInst = store.getTodos(dateKeyOf(wed))[0]
  store.removeTodo(dateKeyOf(wed), todayInst.id)
  assert.equal(store.applyRepeats(wed), 0)
  const thu = new Date('2026-08-13T12:00:00')
  assert.equal(store.applyRepeats(thu), 1)
  const reloaded = new DataStore(path.join(dir, 'data.json'))
  assert.ok(Object.keys(reloaded.data.repeatSkips).length > 0)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('当天创建的每日任务被删除后，次日仍会生成', () => {
  const { store, dir } = tmpStore()
  const today = new Date('2026-08-12T12:00:00')
  const tpl = store.addTodo(dateKeyOf(today), '晨跑', { repeat: 'daily' }).at(-1)
  store.removeTodo(dateKeyOf(today), tpl.id)
  const tomorrow = new Date('2026-08-13T12:00:00')
  assert.equal(store.applyRepeats(tomorrow), 1)
  assert.equal(store.getTodos(dateKeyOf(tomorrow))[0].text, '晨跑')
  fs.rmSync(dir, { recursive: true, force: true })
})

test('重复模板在创建时登记，删除实例不丢失模板', () => {
  const { store, dir } = tmpStore()
  const today = new Date('2026-08-12T12:00:00')
  const tpl = store.addTodo(dateKeyOf(today), '喝水', { repeat: 'daily', priority: 'high' }).at(-1)
  assert.ok(store.data.repeatTemplates[tpl.id])
  store.removeTodo(dateKeyOf(today), tpl.id)
  const tomorrow = new Date('2026-08-13T12:00:00')
  store.applyRepeats(tomorrow)
  const inst = store.getTodos(dateKeyOf(tomorrow))[0]
  assert.equal(inst.priority, 'high')
  fs.rmSync(dir, { recursive: true, force: true })
})

test('每日任务的完成状态每天独立，次日副本未完成', () => {
  const { store, dir } = tmpStore()
  const tue = new Date('2026-08-11T12:00:00')
  store.addTodo(dateKeyOf(tue), '晨跑', { repeat: 'daily' })
  const wed = new Date('2026-08-12T12:00:00')
  store.applyRepeats(wed)
  const wedInst = store.getTodos(dateKeyOf(wed))[0]
  store.toggleTodo(dateKeyOf(wed), wedInst.id)
  assert.equal(store.getTodos(dateKeyOf(wed))[0].done, true)
  const thu = new Date('2026-08-13T12:00:00')
  store.applyRepeats(thu)
  const thuInst = store.getTodos(dateKeyOf(thu))[0]
  assert.equal(thuInst.done, false)
  assert.notEqual(thuInst.id, wedInst.id)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('当天创建的每日任务不会当天重复生成，次日才生成', () => {
  const { store, dir } = tmpStore()
  const today = new Date('2026-08-12T12:00:00')
  store.addTodo(dateKeyOf(today), '英语', { repeat: 'daily' })
  assert.equal(store.applyRepeats(today), 0)
  assert.equal(store.getTodos(dateKeyOf(today)).length, 1)
  const tomorrow = new Date('2026-08-13T12:00:00')
  assert.equal(store.applyRepeats(tomorrow), 1)
  assert.equal(store.getTodos(dateKeyOf(tomorrow)).length, 1)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('updateRepeat 修改模板与所有实例的周期', () => {
  const { store, dir } = tmpStore()
  const today = new Date('2026-08-12T12:00:00')
  const tpl = store.addTodo(dateKeyOf(today), '英语', { repeat: 'daily' }).at(-1)
  const tomorrow = new Date('2026-08-13T12:00:00')
  store.applyRepeats(tomorrow)
  assert.equal(store.updateRepeat(tpl.id, 'weekly'), 2)
  assert.equal(store.data.repeatTemplates[tpl.id].repeat, 'weekly')
  assert.equal(store.getTodos(dateKeyOf(today))[0].repeat, 'weekly')
  assert.equal(store.getTodos(dateKeyOf(tomorrow))[0].repeat, 'weekly')
  const nextDay = new Date('2026-08-14T12:00:00')
  assert.equal(store.applyRepeats(nextDay), 0)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('修改为不重复后不再生成', () => {
  const { store, dir } = tmpStore()
  const today = new Date('2026-08-12T12:00:00')
  const tpl = store.addTodo(dateKeyOf(today), '打卡', { repeat: 'daily' }).at(-1)
  store.updateRepeat(tpl.id, 'none')
  const tomorrow = new Date('2026-08-13T12:00:00')
  assert.equal(store.applyRepeats(tomorrow), 0)
  assert.equal(store.getTodos(dateKeyOf(tomorrow)).length, 0)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('deleteRepeatTemplate 删除所有实例后不再生成', () => {
  const { store, dir } = tmpStore()
  const today = new Date('2026-08-12T12:00:00')
  const tpl = store.addTodo(dateKeyOf(today), '健身', { repeat: 'daily' }).at(-1)
  const tomorrow = new Date('2026-08-13T12:00:00')
  store.applyRepeats(tomorrow)
  store.removeTodo(dateKeyOf(today), tpl.id)
  assert.equal(store.deleteRepeatTemplate(tpl.id), 1)
  assert.equal(store.data.repeatTemplates[tpl.id], undefined)
  assert.equal(store.getTodos(dateKeyOf(today)).length, 0)
  assert.equal(store.getTodos(dateKeyOf(tomorrow)).length, 0)
  assert.equal(store.applyRepeats(new Date('2026-08-14T12:00:00')), 0)
  fs.rmSync(dir, { recursive: true, force: true })
})

test('损坏的 data.json 自动恢复默认结构', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'planner-test-'))
  fs.writeFileSync(path.join(dir, 'data.json'), 'not-json{{{')
  const store = new DataStore(path.join(dir, 'data.json'))
  assert.deepEqual(store.getTasks(), [])
  assert.deepEqual(store.getTodos('2026-08-12'), [])
  fs.rmSync(dir, { recursive: true, force: true })
})
