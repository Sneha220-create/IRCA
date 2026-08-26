| ticket_id | category | gold_root_cause | tier1_rate | classification | gold_token_hits | metric_token_hits |
|---|---|---|---|---|---|---|
| T001 | easy | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection | active, connections |
| T002 | easy | memory_cache_overgrowth | 1.0 | leaked_strong | memory, cache, overgrowth | cache, memory |
| T003 | easy | network_ingress_queue_exhaustion | 1.0 | leaked_strong | ingress, queue | ingress, queue, depth |
| T004 | easy | deploy_healthcheck_misconfiguration | 1.0 | leaked_strong | deploy | readiness, probe |
| T005 | easy | auth_signing_key_mismatch | 1.0 | leaked_strong | auth | auth |
| T006 | multi_step | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection, pool | pool, connections |
| T007 | multi_step | memory_cache_overgrowth | 1.0 | leaked_strong | memory, cache | cache, memory |
| T008 | multi_step | deploy_healthcheck_misconfiguration | 0.3333333333333333 | leaked_partial | deploy, healthcheck | readiness, success |
| T009 | multi_step | disk_log_rotation_gap | 1.0 | leaked_strong | disk, rotation | disk |
| T010 | multi_step | auth_signing_key_mismatch | 1.0 | leaked_strong | auth | auth, token |
| T011 | tool_heavy | network_ingress_queue_exhaustion | 1.0 | leaked_strong | ingress, queue | ingress, queue |
| T012 | tool_heavy | disk_log_rotation_gap | 1.0 | leaked_strong | - | usage |
| T013 | tool_heavy | deploy_healthcheck_misconfiguration | 1.0 | leaked_strong | deploy | probe, success |
| T016 | easy | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection, pool | pool, active, connections |
| T017 | easy | memory_cache_overgrowth | 1.0 | leaked_strong | memory, cache | cache, memory |
| T018 | easy | network_ingress_queue_exhaustion | 1.0 | leaked_strong | queue | queue |
| T019 | easy | deploy_healthcheck_misconfiguration | 1.0 | leaked_strong | - | readiness, probe |
| T020 | easy | auth_signing_key_mismatch | 1.0 | leaked_strong | auth, signing | auth |
| T021 | easy | disk_log_rotation_gap | 1.0 | leaked_strong | disk | disk |
| T022 | easy | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection, pool | pool, active, connections |
| T023 | easy | memory_cache_overgrowth | 0.6666666666666666 | leaked_partial | memory, cache | cache, memory |
| T024 | easy | network_ingress_queue_exhaustion | 1.0 | leaked_strong | ingress, queue | ingress, queue, depth |
| T025 | easy | deploy_healthcheck_misconfiguration | 1.0 | leaked_strong | - | probe |
| T026 | multi_step | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection, pool | pool, connections |
| T027 | multi_step | memory_cache_overgrowth | 1.0 | leaked_strong | memory, cache | cache, memory |
| T028 | multi_step | network_ingress_queue_exhaustion | 1.0 | leaked_strong | ingress, queue | ingress, queue, depth |
| T029 | multi_step | deploy_healthcheck_misconfiguration | 1.0 | leaked_strong | - | readiness, probe |
| T030 | multi_step | auth_signing_key_mismatch | 1.0 | leaked_strong | auth | auth |
| T031 | multi_step | disk_log_rotation_gap | 1.0 | leaked_strong | disk | disk |
| T032 | multi_step | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection, pool | pool, connections |
| T033 | multi_step | memory_cache_overgrowth | 0.6666666666666666 | leaked_partial | memory, cache, overgrowth | cache, memory |
| T034 | multi_step | network_ingress_queue_exhaustion | 1.0 | leaked_strong | ingress, queue | ingress, queue, depth |
| T035 | multi_step | deploy_healthcheck_misconfiguration | 1.0 | leaked_strong | deploy, healthcheck | probe |
| T036 | multi_step | auth_signing_key_mismatch | 1.0 | leaked_strong | auth, mismatch | auth, failure |
| T037 | multi_step | disk_log_rotation_gap | 1.0 | leaked_strong | disk | disk |
| T038 | multi_step | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection, pool | pool, connections |
| T039 | multi_step | memory_cache_overgrowth | 1.0 | leaked_strong | memory, cache | cache, memory |
| T040 | multi_step | network_ingress_queue_exhaustion | 1.0 | leaked_strong | ingress, queue | ingress, queue, depth |
| T041 | tool_heavy | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection, pool | pool, connections |
| T042 | tool_heavy | memory_cache_overgrowth | 1.0 | leaked_strong | memory, cache | cache, memory |
| T043 | tool_heavy | network_ingress_queue_exhaustion | 1.0 | leaked_strong | ingress, queue | ingress, queue |
| T044 | tool_heavy | deploy_healthcheck_misconfiguration | 1.0 | leaked_strong | deploy, healthcheck, misconfiguration | readiness, probe |
| T045 | tool_heavy | auth_signing_key_mismatch | 1.0 | leaked_strong | auth, signing, mismatch | auth, failure |
| T046 | tool_heavy | disk_log_rotation_gap | 1.0 | leaked_strong | disk | disk |
| T047 | tool_heavy | db_connection_pool_exhaustion | 1.0 | leaked_strong | pool | pool |
| T048 | rag_heavy | db_connection_pool_exhaustion | 1.0 | leaked_strong | - | - |
| T049 | rag_heavy | memory_cache_overgrowth | 1.0 | leaked_strong | - | - |
| T050 | rag_heavy | network_ingress_queue_exhaustion | 1.0 | leaked_strong | - | - |
| T051 | rag_heavy | deploy_healthcheck_misconfiguration | 0.6666666666666666 | leaked_partial | healthcheck | - |
| T052 | rag_heavy | auth_signing_key_mismatch | 1.0 | leaked_strong | auth | auth, token, verification, failure |
| T053 | rag_heavy | disk_log_rotation_gap | 1.0 | leaked_strong | - | - |
| T054 | rag_heavy | db_connection_pool_exhaustion | 1.0 | leaked_strong | connection | connections |
| T055 | rag_heavy | memory_cache_overgrowth | 1.0 | leaked_strong | - | - |
