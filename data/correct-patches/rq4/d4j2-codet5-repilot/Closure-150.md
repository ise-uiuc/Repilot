# Repilot Patch

```


      super.visit(t, n, parent);
```

# Developer Patch

```
      super.visit(t, n, parent);
```

# Context

```
--- bug/Closure-150/src/com/google/javascript/jscomp/TypedScopeCreator.java

+++ fix/Closure-150/src/com/google/javascript/jscomp/TypedScopeCreator.java

@@ -1448,21 +1448,9 @@

         return;
       }
 
-      attachLiteralTypes(n);
-      switch (n.getType()) {
-        case Token.FUNCTION:
-          if (parent.getType() == Token.NAME) {
-            return;
-          }
-          defineDeclaredFunction(n, parent);
-          break;
-        case Token.CATCH:
-          defineCatch(n, parent);
-          break;
-        case Token.VAR:
-          defineVar(n, parent);
-          break;
-      }
+
+
+      super.visit(t, n, parent);
     }
 
     /** Handle bleeding functions and function parameters. */
```

# Note

