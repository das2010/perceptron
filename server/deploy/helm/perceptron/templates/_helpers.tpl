{{- define "perceptron.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "perceptron.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "perceptron.secretName" -}}
{{- default (printf "%s-secrets" (include "perceptron.fullname" .)) .Values.existingSecret -}}
{{- end -}}

{{- define "perceptron.redisUrl" -}}
{{- if .Values.valkey.enabled -}}
redis://{{ include "perceptron.fullname" . }}-valkey:6379/0
{{- else -}}
{{- required "valkey.externalUrl es obligatorio si valkey.enabled=false" .Values.valkey.externalUrl -}}
{{- end -}}
{{- end -}}

{{- define "perceptron.trackingUri" -}}
{{- if .Values.mlflow.enabled -}}
http://{{ include "perceptron.fullname" . }}-mlflow:5000
{{- else -}}
{{- .Values.mlflow.externalUrl -}}
{{- end -}}
{{- end -}}

{{/* Variables comunes al servidor y a los workers. */}}
{{- define "perceptron.env" -}}
- name: PERCEPTRON_DATABASE_URL
  valueFrom: { secretKeyRef: { name: {{ include "perceptron.secretName" . }}, key: database-url } }
- name: PERCEPTRON_SERVER__SECRET_KEY
  valueFrom: { secretKeyRef: { name: {{ include "perceptron.secretName" . }}, key: secret-key } }
- name: PERCEPTRON_MASTER_KEY
  valueFrom: { secretKeyRef: { name: {{ include "perceptron.secretName" . }}, key: master-key } }
- name: PERCEPTRON_SERVER__REDIS_URL
  value: {{ include "perceptron.redisUrl" . | quote }}
{{- with include "perceptron.trackingUri" . }}
- name: PERCEPTRON_TRACKING_URI
  value: {{ . | quote }}
{{- end }}
- name: PERCEPTRON_SERVER__MAX_RUNNING_STUDIES_PER_USER
  value: {{ .Values.server.quotas.perUser | quote }}
- name: PERCEPTRON_SERVER__MAX_RUNNING_STUDIES_PER_WORKSPACE
  value: {{ .Values.server.quotas.perWorkspace | quote }}
{{- if .Values.sources.existingClaim }}
- name: PERCEPTRON_SOURCE_ROOTS
  value: '["/sources"]'
{{- end }}
{{- end -}}

{{- define "perceptron.volumes" -}}
- name: workspace
  persistentVolumeClaim:
    claimName: {{ default (printf "%s-workspace" (include "perceptron.fullname" .)) .Values.workspace.existingClaim }}
{{- if .Values.sources.existingClaim }}
- name: sources
  persistentVolumeClaim:
    claimName: {{ .Values.sources.existingClaim }}
    readOnly: true
{{- end }}
{{- end -}}

{{- define "perceptron.volumeMounts" -}}
- name: workspace
  mountPath: /data
{{- if .Values.sources.existingClaim }}
- name: sources
  mountPath: /sources
  readOnly: true
{{- end }}
{{- end -}}
