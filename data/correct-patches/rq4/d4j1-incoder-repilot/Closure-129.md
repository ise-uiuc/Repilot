# Repilot Patch

```
      while (first.isCast()) {
        first = first.getFirstChild();
      }
```

# Developer Patch

```
      while (first.isCast()) {
        first = first.getFirstChild();
      }
```

# Context

```
--- bug/Closure-129/src/com/google/javascript/jscomp/PrepareAst.java

+++ fix/Closure-129/src/com/google/javascript/jscomp/PrepareAst.java

@@ -163,6 +163,9 @@

       Node first = n.getFirstChild();
 
       // ignore cast nodes.
+      while (first.isCast()) {
+        first = first.getFirstChild();
+      }
 
       if (!NodeUtil.isGet(first)) {
         n.putBooleanProp(Node.FREE_CALL, true);
```

# Note

