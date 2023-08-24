# Repilot Patch

```

        if (longOpts.keySet().contains(opt))
        {
            return Collections.singletonList(opt);
        }
```

# Developer Patch

```
        if(longOpts.keySet().contains(opt)) {
            return Collections.singletonList(opt);
        }
```

# Context

```
--- bug/Cli-35/src/main/java/org/apache/commons/cli/Options.java

+++ fix/Cli-35/src/main/java/org/apache/commons/cli/Options.java

@@ -238,6 +238,11 @@

 
         // for a perfect match return the single option only
 
+        if (longOpts.keySet().contains(opt))
+        {
+            return Collections.singletonList(opt);
+        }
+
         for (String longOpt : longOpts.keySet())
         {
             if (longOpt.startsWith(opt))
```

# Note

