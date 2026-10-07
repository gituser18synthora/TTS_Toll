// Thin client for the FastAPI backend (same origin; Vite proxies /api in development).

export function qs(params) {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params || {})) {
    if (v === undefined || v === null || v === '') continue
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, x))
    else sp.append(k, v)
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

function formatDetail(detail) {
  if (!detail) return ''
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((e) => `${(e.loc || []).slice(-1)[0]}: ${e.msg}`).join('; ')
  return JSON.stringify(detail)
}

async function request(method, url, body) {
  const opts = { method, headers: {} }
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  let res
  try {
    res = await fetch(url, opts)
  } catch {
    throw new Error('Cannot reach the backend. Is it running?')
  }
  if (!res.ok) {
    let detail = ''
    try {
      detail = formatDetail((await res.json()).detail)
    } catch {
      /* not JSON */
    }
    throw new Error(detail || `${res.status} ${res.statusText}`)
  }
  return res.status === 204 ? null : res.json()
}

export const api = {
  health: () => request('GET', '/api/health'),
  config: () => request('GET', '/api/config'),
  stats: () => request('GET', '/api/stats'),

  speakers: () => request('GET', '/api/speakers'),

  sources: (params) => request('GET', `/api/sources${qs(params)}`),
  updateSource: (id, body) => request('PATCH', `/api/sources/${id}`, body),
  processSource: (id) => request('POST', `/api/sources/${id}/process`),
  reprocessSource: (id) => request('POST', `/api/sources/${id}/reprocess`),
  retranscribeSource: (id) => request('POST', `/api/sources/${id}/retranscribe`),
  retranscribeAll: () => request('POST', '/api/sources/retranscribe-all'),
  sourceSpeakers: (id) => request('GET', `/api/sources/${id}/speakers`),
  assignSpeakers: (id, assignments) => request('POST', `/api/sources/${id}/speakers`, { assignments }),
  speakerCheckAll: () => request('POST', '/api/maintenance/speaker-check'),
  dedupAll: () => request('POST', '/api/maintenance/dedup'),

  clips: (params) => request('GET', `/api/clips${qs(params)}`),
  updateClip: (id, body) => request('PATCH', `/api/clips/${id}`, body),
  bulk: (body) => request('POST', '/api/clips/bulk', body),

  jobs: (params) => request('GET', `/api/jobs${qs(params)}`),
  job: (id) => request('GET', `/api/jobs/${id}`),
  cancelJob: (id) => request('POST', `/api/jobs/${id}/cancel`),
  retryJob: (id) => request('POST', `/api/jobs/${id}/retry`),

  exportPreview: (params) => request('GET', `/api/exports/preview${qs(params)}`),
  exports: () => request('GET', '/api/exports'),
  createExport: (body) => request('POST', '/api/exports', body),
  rerunExport: (id) => request('POST', `/api/exports/${id}/rerun`),
}

// Raw-body upload with progress (fetch cannot report upload progress).
export function uploadFile(file, meta, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `/api/uploads${qs({ ...meta, filename: file.name })}`)
    xhr.setRequestHeader('Content-Type', 'application/octet-stream')
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total)
    xhr.onload = () => {
      let body = null
      try {
        body = JSON.parse(xhr.responseText)
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else reject(new Error(formatDetail(body?.detail) || `${xhr.status} ${xhr.statusText}`))
    }
    xhr.onerror = () => reject(new Error('network error during upload'))
    xhr.send(file)
  })
}
