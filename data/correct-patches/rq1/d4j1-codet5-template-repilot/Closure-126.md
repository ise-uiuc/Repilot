# Repilot Patch

```
if (NodeUtil.hasFinally(n)) {
       
        // noop
      }
```

# Developer Patch

```

```

# Context

```
--- bug/Closure-126/src/com/google/javascript/jscomp/MinimizeExitPoints.java

+++ fix/Closure-126/src/com/google/javascript/jscomp/MinimizeExitPoints.java

@@ -138,9 +138,9 @@

        * can cause problems if it changes the completion type of the finally
        * block. See ECMA 262 Sections 8.9 & 12.14
        */
-      if (NodeUtil.hasFinally(n)) {
-        Node finallyBlock = n.getLastChild();
-        tryMinimizeExits(finallyBlock, exitType, labelName);
+if (NodeUtil.hasFinally(n)) {
+       
+        // noop
       }
     }
```

# Note

