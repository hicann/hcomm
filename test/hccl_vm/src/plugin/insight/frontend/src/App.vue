<script setup>
import { computed, ref } from 'vue'
import AppTopNav from './components/AppTopNav.vue'
import AnalyticPage from './pages/AnalyticPage.vue'
import DashboardPage from './pages/DashboardPage.vue'
import MemViewPage from './pages/MemViewPage.vue'
import CcuTracePage from './pages/CcuTracePage.vue'

const pages = [
  { id: 'dashboard', label: '总览' },
  { id: 'mem-view', label: '关联' },
  { id: 'analytic', label: '报错' },
  { id: 'ccu-trace', label: 'CCU调试' },
]

const activePage = ref('dashboard')

const pageComponents = {
  dashboard: DashboardPage,
  'mem-view': MemViewPage,
  analytic: AnalyticPage,
  'ccu-trace': CcuTracePage,
}

const activeComponent = computed(() => pageComponents[activePage.value] ?? DashboardPage)
</script>

<template>
  <div class="app-shell">
    <AppTopNav :pages="pages" :active-page="activePage" @change="activePage = $event" />
    <main class="app-main">
      <component :is="activeComponent" @navigate-page="activePage = $event" />
    </main>
  </div>
</template>
