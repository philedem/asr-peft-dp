{{/*
Expand the name of the chart.
*/}}
{{- define "asr-peft.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "asr-peft.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "asr-peft.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "asr-peft.labels" -}}
helm.sh/chart: {{ include "asr-peft.chart" . }}
{{ include "asr-peft.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "asr-peft.selectorLabels" -}}
app.kubernetes.io/name: {{ include "asr-peft.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Backend labels
*/}}
{{- define "asr-peft.backend.labels" -}}
{{ include "asr-peft.labels" . }}
app: {{ .Values.backend.name }}
{{- end }}

{{/*
Frontend labels
*/}}
{{- define "asr-peft.frontend.labels" -}}
{{ include "asr-peft.labels" . }}
app: {{ .Values.frontend.name }}
{{- end }}
