(module
  (type (;0;) (func (result i64)))
  (func (;0;) (type 0) (result i64)
    (local i64 i64 i32 i64)
    i64.const 0
    local.set 0
    i64.const 1
    local.set 1
    i32.const 0
    local.set 2
    loop (result i64)  ;; label = @1
      local.get 1
      local.set 3
      block  ;; label = @2
        local.get 2
        i32.const 0
        i32.load offset=1024
        i32.lt_s
        br_if 0 (;@2;)
        local.get 0
        return
      end
      local.get 2
      i32.const 1
      i32.add
      local.set 2
      local.get 0
      local.get 3
      i64.add
      i64.const 1000000007
      i64.rem_s
      local.set 1
      local.get 3
      local.set 0
      br 0 (;@1;)
    end)
  (memory (;0;) 2)
  (global (;0;) (mut i32) (i32.const 66576))
  (export "memory" (memory 0))
  (export "solve" (func 0))
  (data (;0;) (i32.const 1024) "P\00\00\00"))
